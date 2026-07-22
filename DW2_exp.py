"""El Zein・Hu・Verma・提案手法の比較実験スクリプト。"""

import contextlib
import io
import os

import numpy as np
import open3d as o3d

import DW1_ELZ_func as DW1ELZ
import DW1_HU_func as DW1HU
import DW1_VER_func as DW1VER
import DW1_X1_func as DW1X1
import DW2_func as DW2F


# ============================== 共通設定 ==============================
TARGET_PSNR = 60.0
INPUT_FILE = "C:/bun_zipper.ply"
# INPUT_FILE = "C:/dragon.ply"
# INPUT_FILE = "C:/Armadillo.ply"
IMAGE_PATH = "watermark16.bmp"
WATERMARK_SIZE = 16
# KMeans・FCMの初期値を変えて作る埋め込みモデル数
NUM_EMBEDDING_TRIALS = 3
# ノイズ・ランダム/FPSダウンサンプリングの攻撃乱数試行数
NUM_ATTACK_TRIALS = 3
EMBEDDING_SEED_BASE = 42
ATTACK_SEED_BASE = 42
VERBOSE_TRIAL_LOGS = False

# 使用可能: "ElZein", "Hu", "Verma", "Proposed"
# 例: COMPARED_METHODS = ["ElZein", "Proposed"]
COMPARED_METHODS = [
    "ElZein",
    "Hu",
    # "Verma",
    "Proposed",
]

EXPERIMENTS = [
    # ("noise", [1.5, 2.0, 2.5]),
    # ("smoothing", [50, 60]),
    ("reordering", [0.25, 0.5, 0.75, 1.0]),
    # ("cropping", [0.6, 0.5, 0.4, 0.3, 0.2]),
    # ("downsampling", [0.5, 1.0, 1.5, 2.0]),
    # ("visual_quality", [None]),
]

NOISE_MODE = "gaussian"
SMOOTHING_LAMBDA = 0.3
SMOOTHING_K = 6
CROPPING_MODE = "axis"
CROPPING_AXIS = 0
DOWNSAMPLING_MODE = "voxel"

# 提案手法
GRAPH_MODE = "knn"
KNN_K = 6
GRAPH_RADIUS = 0.03
ORDER_CHECK_K = 6                 # 元点群で調べる近傍数
ORDER_CHECK_EDGE_FACTOR = 10.0     # 攻撃後点間隔に対するアウト辺判定倍率
ORDER_CHECK_MAX_BAD_RATIO = 0.005  # アウト辺を許容する最大割合
ORDER_CHECK_MAX_SAMPLES = 20000   # 判定に使う最大頂点数
MATCH_DISTANCE_FACTOR = 10.0       # 座標対応を許容する点間隔倍率
FLATNESS_WEIGHTING = 0
K_NEIGHBORS = 20
CLUSTER_POINTS_PROPOSED = [2000]
BANDS = [
    ("Full", 0.0, 1.0, 1.6e-3),
    ("Low", 0.0, 0.2, 3.6e-3),
    ("High", 0.8, 1.0, 3.6e-3),
]

# Hu手法
HU_INITIAL_FIDEP = 53.0
HU_T = 25
HU_ARNOLD_ITERATIONS = 20


def point_cloud(vertices):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(vertices))
    return pcd


def calculate_target_mse(vertices, target_psnr):
    peak = np.linalg.norm(np.ptp(vertices, axis=0))
    return peak**2 / 10 ** (target_psnr / 10.0)


def embedding_quality(vertices, marked_vertices):
    metrics = DW2F.evaluate_psnr(
        point_cloud(vertices), point_cloud(marked_vertices), by_index=True, verbose=False
    )
    return metrics["mse"], metrics["psnr"]


def apply_attack(vertices, attack_type, parameter, seed):
    if attack_type == "noise":
        return DW2F.noise_addition_attack(
            vertices, noise_percent=parameter, mode=NOISE_MODE, seed=seed
        )
    if attack_type == "smoothing":
        return DW2F.smoothing_attack(
            vertices,
            lambda_val=SMOOTHING_LAMBDA,
            iterations=int(parameter),
            k=SMOOTHING_K,
        )
    if attack_type == "cropping":
        return DW2F.cropping_attack(
            vertices,
            keep_ratio=parameter,
            mode=CROPPING_MODE,
            axis=CROPPING_AXIS,
        )
    if attack_type == "downsampling":
        return DW2F.downsampling_attack(
            vertices,
            mode=DOWNSAMPLING_MODE,
            voxel_size_percent=parameter,
            seed=seed,
        )
    if attack_type == "reordering":
        return DW2F.vertex_reordering_attack(
            vertices, reorder_ratio=parameter, seed=seed
        )
    if attack_type == "visual_quality":
        return np.asarray(vertices).copy()
    raise ValueError(f"Unknown experiment type: {attack_type}")


def run_trial_call(function, *args, **kwargs):
    """試行内の通常ログを抑えつつ、例外は呼び出し元へ伝える。"""
    if VERBOSE_TRIAL_LOGS:
        return function(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def match_strength(
    original,
    embed_function,
    target_mse,
    initial_strength,
    label,
    return_strength=False,
):
    """変位と強度の比例関係から、目標MSEに対応する強度を求める。"""
    candidate = embed_function(initial_strength)
    mse, _ = embedding_quality(original, candidate)
    if mse <= 0.0 or not np.isfinite(mse):
        raise RuntimeError(f"{label}: initial embedding produced invalid MSE={mse}.")

    strength = initial_strength * np.sqrt(target_mse / mse)
    candidate = embed_function(strength)
    mse, psnr = embedding_quality(original, candidate)

    # 固有分解や再構成の数値誤差が見える場合だけ一度補正する。
    if mse > 0.0 and abs(mse - target_mse) / target_mse >= 3e-4:
        strength *= np.sqrt(target_mse / mse)
        candidate = embed_function(strength)
        mse, psnr = embedding_quality(original, candidate)

    print(f"[{label}] strength={strength:.6e}, PSNR={psnr:.2f} dB")
    if return_strength:
        return candidate, strength
    return candidate


def match_hu_fidep(vertices, triangles, bits, target_mse):
    """Hu法のFidePを調整し、埋め込み後の実測MSEを目標値に合わせる。"""

    def embed(fidep):
        return DW1HU.embed_watermark_hu_mesh(
            vertices,
            triangles,
            bits,
            FideP=fidep,
            T=HU_T,
            watermark_size=WATERMARK_SIZE,
            arnold_iterations=HU_ARNOLD_ITERATIONS,
        )

    marked, key, alpha = embed(HU_INITIAL_FIDEP)
    mse, _ = embedding_quality(vertices, marked)
    if mse <= 0.0:
        raise RuntimeError("Hu: initial embedding produced zero MSE.")

    # alpha is proportional to 10**(-FideP/20), while MSE is proportional
    # to alpha**2. Therefore the target FideP can be calculated directly.
    fidep = HU_INITIAL_FIDEP - 10.0 * np.log10(target_mse / mse)
    marked, key, alpha = embed(fidep)
    mse, psnr = embedding_quality(vertices, marked)

    # Correct once if reconstruction introduced a measurable numerical error.
    if abs(mse - target_mse) / target_mse >= 3e-4 and mse > 0.0:
        fidep -= 10.0 * np.log10(target_mse / mse)
        marked, key, alpha = embed(fidep)
        mse, psnr = embedding_quality(vertices, marked)

    print(
        f"[Hu] FideP={fidep:.6f}, alpha={alpha:.6e}, "
        f"PSNR={psnr:.2f} dB"
    )
    return marked, key, alpha


def embed_proposed(vertices, labels, bits, min_spectrum, max_spectrum, beta):
    return DW1X1.embed_watermark_x1(
        vertices,
        labels,
        bits,
        beta=beta,
        graph_mode=GRAPH_MODE,
        k=KNN_K,
        radius=GRAPH_RADIUS,
        flatness_weighting=FLATNESS_WEIGHTING,
        k_neighbors=K_NEIGHBORS,
        min_spectre=min_spectrum,
        max_spectre=max_spectrum,
    )


def visual_quality_scores(original, marked):
    """いずれも高いほど高品質となる実装上のスコアを返す。"""
    before = point_cloud(original)
    after = point_cloud(marked)
    return {
        "PC-MSDM": DW2F.evaluate_pc_msdm(before, after, verbose=False),
        "AngularSimilarity": DW2F.evaluate_angular_similarity(
            before, after, verbose=False
        ),
        "P2D": DW2F.evaluate_p2d(before, after, verbose=False),
        "PointSSIM": DW2F.evaluate_point_ssim(before, after, verbose=False),
    }


def bit_error_rate(reference_bits, extracted_bits):
    _, ber = DW2F.evaluate_robustness(
        reference_bits, extracted_bits, verbose=False
    )
    return ber


def load_mesh():
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"Input mesh not found: {INPUT_FILE}")
    mesh = o3d.io.read_triangle_mesh(INPUT_FILE)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("The comparison requires a triangle mesh (PLY/OFF/OBJ).")

    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles).copy()
    vertices, triangles, _ = DW2F.remove_unreferenced_vertices(vertices, triangles)
    pcd = DW2F.normalize_point_cloud(point_cloud(vertices))
    return np.asarray(pcd.points).copy(), triangles


def prepare_methods(
    vertices,
    triangles,
    bits,
    target_mse,
    embedding_seed,
    deterministic_cache,
):
    """各方式を一度埋め込み、方式名・頂点・抽出関数をまとめる。"""
    methods = {}
    newly_prepared = []
    available = {"ElZein", "Hu", "Verma", "Proposed"}
    selected_methods = set(COMPARED_METHODS)
    unknown = selected_methods - available
    if unknown:
        raise ValueError(
            f"Unknown method(s): {sorted(unknown)}. "
            f"Available methods: {sorted(available)}"
        )
    if not selected_methods:
        raise ValueError("COMPARED_METHODS must contain at least one method.")

    if "ElZein" in selected_methods:
        print("\n[ElZein] redundant Portion 2 embedding...")
        elzein_selected = DW1ELZ.local_feature_clustering(
            vertices, triangles, seed=embedding_seed, verbose=False
        )
        _, elzein_strength = match_strength(
            vertices,
            lambda a: DW1ELZ.embed_watermark_elzein_mesh(
                vertices,
                triangles,
                bits,
                a=a,
                verbose=False,
                seed=embedding_seed,
                carrier_indices=elzein_selected,
            )[0],
            target_mse,
            1e-3,
            "ElZein",
            return_strength=True,
        )
        marked, elzein_key = DW1ELZ.embed_watermark_elzein_mesh(
            vertices,
            triangles,
            bits,
            a=elzein_strength,
            verbose=False,
            seed=embedding_seed,
            carrier_indices=elzein_selected,
        )
        methods["ElZein"] = {
            "vertices": marked,
            "extract": lambda attacked: (
                DW1ELZ.extract_watermark_elzein_mesh(
                    attacked,
                    vertices,
                    triangles,
                    key_info=elzein_key,
                    verbose=False,
                )
            ),
        }
        newly_prepared.append("ElZein")

    if "Hu" in selected_methods:
        if "Hu" in deterministic_cache:
            methods["Hu"] = deterministic_cache["Hu"]
        else:
            print("\n[Hu] surface EMD embedding...")
            hu_marked, hu_key, _ = match_hu_fidep(
                vertices, triangles, bits, target_mse
            )
            methods["Hu"] = {
                "vertices": hu_marked,
                "extract": lambda attacked: (
                    DW1HU.extract_watermark_hu_mesh(attacked, triangles, hu_key)
                ),
            }
            deterministic_cache["Hu"] = methods["Hu"]
            newly_prepared.append("Hu")

    if "Verma" in selected_methods:
        if "Verma" in deterministic_cache:
            methods["Verma"] = deterministic_cache["Verma"]
        else:
            print("\n[Verma] virtual histogram embedding...")
            verma_marked, verma_key, _ = DW1VER.embed_watermark_verma_mesh(
                vertices, triangles, bits
            )
            methods["Verma"] = {
                "vertices": verma_marked,
                "extract": lambda attacked: DW1VER.extract_watermark_verma_mesh(
                    vertices,
                    attacked,
                    triangles,
                    key_info=verma_key,
                ),
            }
            deterministic_cache["Verma"] = methods["Verma"]
            newly_prepared.append("Verma")

    if "Proposed" in selected_methods:
        for cluster_points in CLUSTER_POINTS_PROPOSED:
            print(
                f"\n[Proposed] KMeans: approximately "
                f"{cluster_points} points/cluster..."
            )
            labels = DW1X1.kmeans_cluster_points(
                vertices, cluster_point=cluster_points, seed=embedding_seed
            )
            min_size = min(
                np.count_nonzero(labels == label) for label in np.unique(labels)
            )
            if min_size < len(bits):
                print(
                    f"[Warning] smallest cluster={min_size}, "
                    f"watermark bits={len(bits)}."
                )
            for band, min_spectrum, max_spectrum, initial_beta in BANDS:
                name = f"P-{band}({cluster_points})"
                marked = match_strength(
                    vertices,
                    lambda beta, lab=labels, lo=min_spectrum, hi=max_spectrum: embed_proposed(
                        vertices, lab, bits, lo, hi, beta
                    ),
                    target_mse,
                    initial_beta,
                    name,
                )
                methods[name] = {
                    "vertices": marked,
                    "extract": lambda attacked, lab=labels, lo=min_spectrum, hi=max_spectrum: (
                        DW1X1.extract_watermark_x1(
                            attacked,
                            vertices,
                            lab,
                            len(bits),
                            graph_mode=GRAPH_MODE,
                            k=KNN_K,
                            radius=GRAPH_RADIUS,
                            min_spectre=lo,
                            max_spectre=hi,
                            match_distance_factor=MATCH_DISTANCE_FACTOR,
                            order_check_k=ORDER_CHECK_K,
                            order_check_edge_factor=ORDER_CHECK_EDGE_FACTOR,
                            order_check_max_bad_ratio=ORDER_CHECK_MAX_BAD_RATIO,
                            order_check_max_samples=ORDER_CHECK_MAX_SAMPLES,
                        )
                    ),
                }
                newly_prepared.append(name)

    print(f"\nEmbedding quality (seed={embedding_seed}, before attacks):")
    for name in newly_prepared:
        method = methods[name]
        _, psnr = embedding_quality(vertices, method["vertices"])
        method["psnr"] = psnr
        print(f"  {name:<20} PSNR={psnr:.2f} dB")
        if name == "Verma":
            print(f"    (fixed strength: TARGET_PSNR is ignored for {name})")
        elif not np.isfinite(psnr) or abs(psnr - TARGET_PSNR) > 0.05:
            raise RuntimeError(
                f"{name}: embedding PSNR is {psnr:.4f} dB; "
                f"expected {TARGET_PSNR:.4f} +/- 0.05 dB."
            )
    return methods, newly_prepared


def verify_no_attack_extraction(methods, bits):
    """攻撃前の埋め込み済みモデルから全方式が完全抽出できることを確認する。"""
    print("\nNo-attack extraction check:")
    for name, method in methods.items():
        extracted = run_trial_call(method["extract"], method["vertices"])
        ber = bit_error_rate(bits, extracted)
        print(f"  {name:<20} BER={ber:.4f}")
        if ber != 0.0:
            raise RuntimeError(
                f"{name}: no-attack BER must be 0 before robustness experiments; "
                f"got {ber:.6f}."
            )


def attack_trial_count(attack_type):
    """乱数を使う攻撃のみ複数回試行し、決定論的攻撃の重複実行を避ける。"""
    if NUM_ATTACK_TRIALS < 1:
        raise ValueError("NUM_ATTACK_TRIALS must be at least 1.")
    if attack_type in {"noise", "reordering"}:
        return NUM_ATTACK_TRIALS
    if attack_type == "downsampling" and DOWNSAMPLING_MODE in {"random", "fps"}:
        return NUM_ATTACK_TRIALS
    return 1


def run_robustness_experiment(method_variants, bits, attack_type, parameters):
    results = {name: [] for name in method_variants}
    trial_count = attack_trial_count(attack_type)
    for parameter in parameters:
        print(f"  {attack_type}={parameter}: {trial_count} attack trial(s)")
        for name, variants in method_variants.items():
            ber_values = []
            for variant_index, method in enumerate(variants):
                for attack_trial in range(trial_count):
                    seed = ATTACK_SEED_BASE + attack_trial
                    try:
                        attacked = run_trial_call(
                            apply_attack,
                            method["vertices"],
                            attack_type,
                            parameter,
                            seed,
                        )
                        extracted = run_trial_call(method["extract"], attacked)
                        ber_values.append(bit_error_rate(bits, extracted))
                    except Exception as error:
                        raise RuntimeError(
                            f"{name}: {attack_type}={parameter}, "
                            f"embedding trial={variant_index + 1}, "
                            f"attack trial={attack_trial + 1} failed."
                        ) from error
            results[name].append(float(np.mean(ber_values)))
    return results


def run_visual_quality_experiment(method_variants, original):
    results = {}
    for name, variants in method_variants.items():
        scores = [
            visual_quality_scores(original, method["vertices"])
            for method in variants
        ]
        results[name] = {
            metric: float(np.mean([score[metric] for score in scores]))
            for metric in scores[0]
        }
    return results


def result_method_label(name, method_variants):
    """固定強度方式の結果ラベルに実測PSNRを付ける。"""
    if name == "Verma":
        return f"{name} ({method_variants[name][0]['psnr']:.2f} dB)"
    return name


def print_robustness_table(
    attack_type, parameters, results, method_variants
):
    names = list(results)
    method_labels = [
        result_method_label(name, method_variants) for name in names
    ]
    parameter_labels = [str(parameter) for parameter in parameters]
    method_width = max(20, *(len(label) for label in method_labels))
    value_width = max(12, *(len(label) for label in parameter_labels))
    print(
        f"\nRobustness: {attack_type}, BER, "
        f"attack trials={attack_trial_count(attack_type)}"
    )
    print(
        "Embedding variants: "
        + ", ".join(
            f"{name}={len(method_variants[name])}" for name in names
        )
    )
    header = [f"{'Method':<{method_width}}"]
    header.extend(f"{label:>{value_width}}" for label in parameter_labels)
    print(" | ".join(header))
    print("-" * (method_width + (value_width + 3) * len(parameters)))
    for name, method_label in zip(names, method_labels):
        row = [f"{method_label:<{method_width}}"]
        row.extend(f"{value:>{value_width}.4f}" for value in results[name])
        print(" | ".join(row))


def print_visual_quality_table(results, method_variants):
    metrics = ("PC-MSDM", "AngularSimilarity", "P2D", "PointSSIM")
    print("\nVisual quality without attack (higher is better)")
    print(" | ".join(f"{value:<20}" for value in ("Method", *metrics)))
    print("-" * 115)
    for name, scores in results.items():
        values = [
            result_method_label(name, method_variants),
            *(f"{scores[metric]:.4f}" for metric in metrics),
        ]
        print(" | ".join(f"{value:<20}" for value in values))


def prepare_method_variants(vertices, triangles, bits, target_mse):
    """乱数依存方式はseedごとに再埋め込みし、決定論的方式は再利用する。"""
    if NUM_EMBEDDING_TRIALS < 1:
        raise ValueError("NUM_EMBEDDING_TRIALS must be at least 1.")
    deterministic_cache = {}
    method_variants = {}
    for trial in range(NUM_EMBEDDING_TRIALS):
        embedding_seed = EMBEDDING_SEED_BASE + trial
        print(
            f"\n========== Embedding trial {trial + 1} / "
            f"{NUM_EMBEDDING_TRIALS} (seed={embedding_seed}) =========="
        )
        np.random.seed(embedding_seed)
        methods, newly_prepared = prepare_methods(
            vertices,
            triangles,
            bits,
            target_mse,
            embedding_seed,
            deterministic_cache,
        )
        new_methods = {name: methods[name] for name in newly_prepared}
        verify_no_attack_extraction(new_methods, bits)
        for name in newly_prepared:
            method_variants.setdefault(name, []).append(methods[name])

    if not method_variants:
        raise RuntimeError("No method variants were prepared.")
    return method_variants


def main():
    np.random.seed(EMBEDDING_SEED_BASE)
    vertices, triangles = load_mesh()
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Watermark image not found: {IMAGE_PATH}")
    bits = DW2F.image_to_bitarray(IMAGE_PATH, n=WATERMARK_SIZE)

    target_mse = calculate_target_mse(vertices, TARGET_PSNR)
    print(
        f"Mesh: {len(vertices)} vertices, {len(triangles)} faces; "
        f"watermark={len(bits)} bits; target PSNR={TARGET_PSNR} dB"
    )
    method_variants = prepare_method_variants(
        vertices, triangles, bits, target_mse
    )

    for experiment_type, parameters in EXPERIMENTS:
        if experiment_type == "visual_quality":
            print_visual_quality_table(
                run_visual_quality_experiment(method_variants, vertices),
                method_variants,
            )
        else:
            results = run_robustness_experiment(
                method_variants, bits, experiment_type, parameters
            )
            print_robustness_table(
                experiment_type, parameters, results, method_variants
            )


if __name__ == "__main__":
    main()
