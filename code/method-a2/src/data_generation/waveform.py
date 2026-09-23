"""由 OpenDSS 相量锚点生成动态电压相量窗口。"""

import numpy as np


def build_dynamic_window(
    pre_phasor: np.ndarray,
    post_phasor: np.ndarray,
    fs: float = 200.0,
    f0: float = 50.0,
    pre_cycles: float = 1.0,
    post_cycles: float = 2.0,
    rng=None,
    perturbation_scale: float = 1.0,
    fault_delay_steps: int = 0,
) -> np.ndarray:
    """生成 `[N,T,6]` 的实部/虚部动态窗口。

    输出通道固定为 `[Re_A, Im_A, Re_B, Im_B, Re_C, Im_C]`；输入相量仍为
    `[幅值_A, 幅值_B, 幅值_C, 角度_A_deg, 角度_B_deg, 角度_C_deg]`。
    """
    if pre_phasor.shape != post_phasor.shape or pre_phasor.ndim != 2:
        raise ValueError("预故障和故障后相量必须具有相同的二维形状")
    if pre_phasor.shape[1] < 6:
        raise ValueError("相量至少需要六个通道")
    if perturbation_scale < 0:
        raise ValueError("perturbation_scale 不得为负数")
    rng = np.random.default_rng() if rng is None else rng
    n_nodes = pre_phasor.shape[0]
    pre_len = max(1, int(round(fs / f0 * pre_cycles)))
    post_len = max(1, int(round(fs / f0 * post_cycles)))
    total_len = pre_len + post_len
    if not 0 <= int(fault_delay_steps) < post_len:
        raise ValueError("fault_delay_steps 必须位于故障后窗口范围内")
    polar = np.zeros((n_nodes, total_len, 6), dtype=np.float32)

    pre_mag = pre_phasor[:, :3]
    post_mag = post_phasor[:, :3]
    pre_ang = np.deg2rad(pre_phasor[:, 3:6])
    post_ang = np.deg2rad(post_phasor[:, 3:6])
    scale = float(perturbation_scale)
    phase_shift = scale * rng.uniform(-0.02, 0.02, size=(n_nodes, 3))
    mag_noise_scale = scale * (0.001 + 0.002 * rng.random((n_nodes, 3)))
    damp_amp = scale * (0.01 + 0.03 * rng.random((n_nodes, 3)))
    t = np.arange(total_len, dtype=np.float64) / fs
    tau = max(0.001, (post_len / fs) * 0.25)
    tau_damp = max(0.001, (post_len / fs) * 0.4)
    fault_index = pre_len + int(fault_delay_steps)
    fault_start = fault_index / fs

    for phase in range(3):
        for idx in range(pre_len):
            perturb = mag_noise_scale[:, phase] * np.sin(
                2.0 * np.pi * 2.0 * t[idx] + phase_shift[:, phase]
            )
            polar[:, idx, phase] = pre_mag[:, phase] * (1.0 + perturb)
            polar[:, idx, 3 + phase] = np.rad2deg(
                pre_ang[:, phase] + phase_shift[:, phase]
            )
        for idx in range(pre_len, total_len):
            if idx < fault_index:
                perturb = mag_noise_scale[:, phase] * np.sin(
                    2.0 * np.pi * 2.0 * t[idx] + phase_shift[:, phase]
                )
                polar[:, idx, phase] = pre_mag[:, phase] * (1.0 + perturb)
                polar[:, idx, 3 + phase] = np.rad2deg(
                    pre_ang[:, phase] + phase_shift[:, phase]
                )
            else:
                elapsed = idx - fault_index
                transition = float(np.exp(-(elapsed + 1) / fs / tau))
                damping = float(np.exp(-elapsed / fs / tau_damp))
                oscillation = damp_amp[:, phase] * damping * np.sin(
                    2.0 * np.pi * f0 * (t[idx] - fault_start) + phase_shift[:, phase]
                )
                polar[:, idx, phase] = (
                    post_mag[:, phase]
                    + (pre_mag[:, phase] - post_mag[:, phase]) * transition
                    + oscillation
                )
                polar[:, idx, 3 + phase] = np.rad2deg(
                    post_ang[:, phase]
                    + (pre_ang[:, phase] - post_ang[:, phase]) * transition
                    + oscillation * 0.5
                )
    magnitude = polar[:, :, :3]
    angle = np.deg2rad(polar[:, :, 3:6])
    real = magnitude * np.cos(angle)
    imag = magnitude * np.sin(angle)
    out = np.empty_like(polar)
    out[:, :, 0::2] = real
    out[:, :, 1::2] = imag
    return out.astype(np.float32)
