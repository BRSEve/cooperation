import math
import numpy as np


class MaritimeChannelModel(object):
    """
    Maritime over-the-sea physical-layer model based on:
    - geometric propagation delay
    - two-ray sea-surface path loss
    - atmospheric / rain attenuation
    - Shannon capacity
    - BER/PER derived from instantaneous SNR
    """

    C = 299792458.0
    K_BOLTZMANN_DBM_HZ = -174.0

    def __init__(self, config=None):
        cfg = config or {}
        self.enable = bool(int(cfg.get("enable", 1)))
        self.frequency_hz = float(cfg.get("carrier_frequency_hz", 1.8e9))
        self.bandwidth_hz = float(cfg.get("bandwidth_hz", 5e6))
        self.tx_power_dbm = float(cfg.get("tx_power_dbm", 43.0))
        self.tx_gain_dbi = float(cfg.get("tx_gain_dbi", 8.0))
        self.rx_gain_dbi = float(cfg.get("rx_gain_dbi", 8.0))
        self.noise_figure_db = float(cfg.get("rx_noise_figure_db", 6.0))
        self.system_loss_db = float(cfg.get("system_loss_db", 2.0))
        self.receiver_sensitivity_dbm = float(cfg.get("receiver_sensitivity_dbm", -95.0))
        self.min_snr_db = float(cfg.get("min_snr_db", 3.0))
        self.snr_gap_db = float(cfg.get("snr_gap_db", 3.0))
        self.packet_size_bits = float(cfg.get("packet_size_bits", 12000.0))
        self.slot_duration_s = float(cfg.get("slot_duration_s", 0.002))
        self.grid_unit_km = float(cfg.get("grid_unit_km", 8.0))
        self.max_edge_delay_steps = int(cfg.get("max_edge_delay_steps", 100))
        self.tx_antenna_height_m = float(cfg.get("tx_antenna_height_m", 25.0))
        self.rx_antenna_height_m = float(cfg.get("rx_antenna_height_m", 25.0))
        self.sea_reflection_magnitude = float(cfg.get("sea_reflection_magnitude", 0.92))
        self.atmospheric_loss_db_per_km = float(cfg.get("atmospheric_loss_db_per_km", 0.01))
        self.rain_loss_db_per_km = float(cfg.get("rain_loss_db_per_km", 0.0))
        self.evaporation_duct_gain_db = float(cfg.get("evaporation_duct_gain_db", 0.0))
        self.wave_height_m = float(cfg.get("significant_wave_height_m", 1.5))

        dynamics = cfg.get("dynamics", {})
        self.dynamic_mode = dynamics.get("mode", "none")
        self.sin_amplitude_db = float(dynamics.get("sin_amplitude_db", 1.5))
        self.sin_period_steps = max(1, int(dynamics.get("sin_period_steps", 20)))
        self.random_walk_sigma_db = float(dynamics.get("random_walk_sigma_db", 0.25))
        self.random_walk_clip_db = float(dynamics.get("random_walk_clip_db", 3.0))
        self.seed = dynamics.get("seed", cfg.get("seed", 2026))
        self._rng = np.random.default_rng(self.seed)
        self._dynamic_margin_db = 0.0
        self._step = 0

    @property
    def wavelength_m(self):
        return self.C / max(self.frequency_hz, 1.0)

    def advance(self, mode=None):
        mode = mode if mode is not None else self.dynamic_mode
        self._step += 1
        if mode == "sinusoidal":
            phase = 2.0 * math.pi * (self._step % self.sin_period_steps) / float(self.sin_period_steps)
            self._dynamic_margin_db = self.sin_amplitude_db * math.sin(phase)
        elif mode == "random_walk":
            self._dynamic_margin_db += float(self._rng.normal(0.0, self.random_walk_sigma_db))
            self._dynamic_margin_db = float(
                max(-self.random_walk_clip_db, min(self.random_walk_clip_db, self._dynamic_margin_db))
            )
        else:
            self._dynamic_margin_db = 0.0

    def _distance_km(self, pos_a, pos_b):
        dx = float(pos_a[0]) - float(pos_b[0])
        dy = float(pos_a[1]) - float(pos_b[1])
        return math.sqrt(dx * dx + dy * dy) * self.grid_unit_km

    def _sea_roughness_factor(self, distance_m):
        grazing = math.atan2(self.tx_antenna_height_m + self.rx_antenna_height_m, max(distance_m, 1e-6))
        sigma_h = max(0.0, self.wave_height_m / 4.0)
        exponent = -0.5 * ((4.0 * math.pi * sigma_h * math.sin(grazing)) / max(self.wavelength_m, 1e-9)) ** 2
        exponent = max(exponent, -60.0)
        return math.exp(exponent)

    def _two_ray_path_loss_db(self, distance_m):
        if distance_m <= 0.0:
            return 0.0

        d_direct = math.sqrt(distance_m ** 2 + (self.tx_antenna_height_m - self.rx_antenna_height_m) ** 2)
        d_reflect = math.sqrt(distance_m ** 2 + (self.tx_antenna_height_m + self.rx_antenna_height_m) ** 2)
        roughness = self._sea_roughness_factor(distance_m)
        reflection_mag = self.sea_reflection_magnitude * roughness
        phase = 2.0 * math.pi * (d_reflect - d_direct) / max(self.wavelength_m, 1e-9)

        field_direct = 1.0 / max(d_direct, 1e-9)
        field_reflect = reflection_mag * complex(math.cos(phase + math.pi), math.sin(phase + math.pi)) / max(d_reflect, 1e-9)
        channel_gain = ((self.wavelength_m / (4.0 * math.pi)) ** 2) * abs(field_direct + field_reflect) ** 2
        channel_gain = max(channel_gain, 1e-18)
        return -10.0 * math.log10(channel_gain)

    def compute_link_metrics(self, pos_a, pos_b):
        distance_km = self._distance_km(pos_a, pos_b)
        distance_m = max(distance_km * 1000.0, 1e-3)

        path_loss_db = self._two_ray_path_loss_db(distance_m)
        path_loss_db += (self.atmospheric_loss_db_per_km + self.rain_loss_db_per_km) * distance_km
        path_loss_db += self.system_loss_db
        path_loss_db += self._dynamic_margin_db
        path_loss_db -= self.evaporation_duct_gain_db

        rx_power_dbm = self.tx_power_dbm + self.tx_gain_dbi + self.rx_gain_dbi - path_loss_db
        noise_dbm = self.K_BOLTZMANN_DBM_HZ + 10.0 * math.log10(max(self.bandwidth_hz, 1.0)) + self.noise_figure_db
        snr_db = rx_power_dbm - noise_dbm
        snr_linear = 10.0 ** (snr_db / 10.0)
        snr_gap_linear = 10.0 ** (self.snr_gap_db / 10.0)
        eff_snr_linear = max(snr_linear / max(snr_gap_linear, 1e-9), 1e-12)
        spectral_efficiency = math.log2(1.0 + eff_snr_linear)
        capacity_bps = self.bandwidth_hz * spectral_efficiency

        ber = 0.5 * math.erfc(math.sqrt(eff_snr_linear))
        ber = min(max(ber, 0.0), 0.5)
        per = 1.0 - ((1.0 - ber) ** self.packet_size_bits)
        per = min(max(per, 0.0), 1.0)

        propagation_delay_s = distance_m / self.C
        serialization_delay_s = self.packet_size_bits / max(capacity_bps, 1.0)
        total_delay_s = propagation_delay_s + serialization_delay_s
        edge_delay = int(math.ceil(total_delay_s / max(self.slot_duration_s, 1e-6)))
        edge_delay = max(1, min(self.max_edge_delay_steps, edge_delay))

        link_available = int((rx_power_dbm >= self.receiver_sensitivity_dbm) and (snr_db >= self.min_snr_db))
        if not link_available:
            per = 1.0
            capacity_bps = 0.0
            edge_delay = self.max_edge_delay_steps

        packets_per_slot = max(1, int(math.floor(max(capacity_bps, self.packet_size_bits) * self.slot_duration_s / self.packet_size_bits)))

        return {
            "distance_km": float(distance_km),
            "path_loss_db": float(path_loss_db),
            "rx_power_dbm": float(rx_power_dbm),
            "noise_dbm": float(noise_dbm),
            "snr_db": float(snr_db),
            "capacity_bps": float(capacity_bps),
            "ber": float(ber),
            "per": float(per),
            "propagation_delay_s": float(propagation_delay_s),
            "serialization_delay_s": float(serialization_delay_s),
            "total_delay_s": float(total_delay_s),
            "edge_delay": int(edge_delay),
            "packets_per_slot": int(packets_per_slot),
            "link_available": int(link_available),
        }

    def apply_to_graph(self, graph):
        if not self.enable:
            return
        for src, dst in graph.edges():
            pos_src = graph.nodes[src].get("position", [0.0, 0.0])
            pos_dst = graph.nodes[dst].get("position", [0.0, 0.0])
            metrics = self.compute_link_metrics(pos_src, pos_dst)
            for key, value in metrics.items():
                graph[src][dst][key] = value
            if "initial_weight" not in graph[src][dst]:
                graph[src][dst]["initial_weight"] = graph[src][dst]["edge_delay"]
