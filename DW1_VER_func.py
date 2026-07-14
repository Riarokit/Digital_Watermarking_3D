"""Verma and Borah (2026) の仮数部ヒストグラムに基づく 3D メッシュデータ隠蔽法.

論文:
    A. Verma and B. Borah,
    "A High Capacity Non-Blind 3D Mesh Data Hiding Technique",
    ICTCon 2024, CCIS 2717, pp. 294-307, 2026.

本ファイルは、論文の式 (1), (2) と Fig. 4, Fig. 5 を基に、公開コードのない
提案法を関数形式で再現したものである。処理の中心は次の通り。

1. 各頂点座標を小数第 6 位に丸める。
2. 各座標の小数 6 桁を 2 桁ずつに分割し、1 頂点あたり 9 個の値を持つ
   2-digit matrix M (shape: N x 9) を作る。
3. M の 00-99 ヒストグラムから最頻値（peak bin）を求める。
4. 秘密ビット列を 3 bit ごとに 0-7 の十進数字へ変換し、2 個ずつ連結して
   00-77 の 2 桁値を作る。従って peak 位置 1 個につき 6 bit を格納する。
5. M を行優先で走査し、元の M が peak bin である位置を秘密値で置換する。
6. 非ブラインド抽出では元メッシュから同じ peak 位置を特定し、透かし入り
   メッシュの対応値を読み出して 6 bit ずつ復元する。

重要な再現上の判断:
- 論文本文は「3 bit ごとに十進変換」と記す一方、Fig. 4 は 3,1,4,5,... を
  31,45,... と 2 個ずつ連結して peak 位置へ格納している。本実装は図と
  Table 2 の容量（payload / 6 が peak 数に対応）に整合する 6 bit/position
  として実装する。
- peak bin が同率の場合の規則は論文にないため、最小の bin を選ぶ。
- 埋め込み位置は Fig. 4 に従い M の行優先順で全 peak 出現箇所を用いる。
- 論文には秘密長の格納方法がないため、末尾 0 padding を除去するための
  ``VermaWatermarkKey`` に元ビット長を保持する。抽出位置自体は鍵ではなく
  元メッシュから再計算する。
- 回転、平行移動、頂点並べ替えに対する位置合わせ手順は論文本文に示されて
  いないため、本実装の基本抽出は元メッシュと透かし入りメッシュで同一の
  頂点順序・座標系を前提とする。
- 論文は 9 列のどの桁対も置換対象としている。この記述をそのまま実装すると
  小数第 1--2 桁の置換で大きな変位が生じ得るため、Table 2 の 103.5 dB 以上の
  PSNR は本文だけからは再現できない。桁を限定する等の未記載処理は加えない。
- 品質・ロバスト性評価は比較実験で定義を統一するため、このファイルには
  実装せず ``DW2_func.py`` の評価関数を使用する。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


_DECIMAL_PLACES = 6
_PAIR_BASE = 100
_SYMBOL_BITS = 3
_SYMBOL_BASE = 1 << _SYMBOL_BITS  # 8
_SYMBOLS_PER_CELL = 2
_BITS_PER_CELL = _SYMBOL_BITS * _SYMBOLS_PER_CELL  # 6
_SCALE = 10 ** _DECIMAL_PLACES


@dataclass(frozen=True)
class VermaWatermarkKey:
    """非ブラインド抽出に必要な最小限の補助情報。

    元メッシュから peak bin と埋め込み位置は再計算できるため、鍵には秘密長と
    使用位置数のみを保存する。``peak_bin`` は再計算結果の検証用である。
    """

    secret_bit_length: int
    used_position_count: int
    peak_bin: int
    capacity_bits: int
    decimal_places: int = _DECIMAL_PLACES
    bits_per_position: int = _BITS_PER_CELL


def _validate_mesh(vertices, triangles):
    """頂点配列と三角形配列を検証し、NumPy 配列として返す。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("vertices must have shape (N, 3) and contain at least one vertex.")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contains NaN or infinity.")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3).")
    if len(triangles) > 0:
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("triangles contains an out-of-range vertex index.")

    return vertices, triangles


def _validate_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not np.isin(bits, (0, 1)).all():
        raise ValueError("secret_bits must contain only 0 and 1.")
    return bits


def remove_unreferenced_vertices(vertices, triangles):
    """三角形面から参照されない孤立頂点を除去し、面番号を再割り当てする。"""
    vertices, triangles = _validate_mesh(vertices, triangles)
    if len(triangles) == 0:
        raise ValueError("triangles is empty; unreferenced vertices cannot be determined.")

    used = np.unique(triangles.ravel())
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return vertices[used].copy(), remap[triangles], used


def find_unreferenced_vertex_indices(vertices, triangles):
    """いずれの三角形面にも現れない頂点番号を返す。"""
    vertices, triangles = _validate_mesh(vertices, triangles)
    if len(triangles) == 0:
        return np.arange(len(vertices), dtype=np.int64)
    used = np.unique(triangles.ravel())
    mask = np.ones(len(vertices), dtype=bool)
    mask[used] = False
    return np.flatnonzero(mask)


def round_vertices_verma(vertices):
    """論文の前処理に従い、全座標を小数第 6 位に丸める。

    浮動小数点文字列に依存せず、絶対値を 10^6 倍した整数で処理する。
    0.5 の完全な tie は NumPy の round-to-even に従うが、通常のメッシュ座標で
    tie が発生することはまれである。
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3).")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contains NaN or infinity.")
    return np.rint(vertices * _SCALE) / _SCALE


def vertices_to_pair_matrix_verma(vertices) -> np.ndarray:
    """頂点座標から論文式 (1), (2) の 2-digit matrix M を生成する。

    返り値の列順は次の通り。
        [x12, x34, x56, y12, y34, y56, z12, z34, z56]

    例:
        (0.121130, -0.991377, 0.128743)
        -> [12, 11, 30, 99, 13, 77, 12, 87, 43]
    """
    rounded = round_vertices_verma(vertices)
    scaled_abs = np.rint(np.abs(rounded) * _SCALE).astype(np.int64)
    mantissa = scaled_abs % _SCALE

    pair_12 = mantissa // 10_000
    pair_34 = (mantissa // 100) % 100
    pair_56 = mantissa % 100

    pairs_per_axis = np.stack((pair_12, pair_34, pair_56), axis=2)  # N x 3 x 3
    return pairs_per_axis.reshape(len(rounded), 9).astype(np.int64)


def pair_matrix_to_vertices_verma(pair_matrix, reference_vertices) -> np.ndarray:
    """変更後の 2-digit matrix を座標へ戻す。

    整数部と符号は ``reference_vertices`` の小数第 6 位丸め後の値から保持し、
    小数 6 桁だけを ``pair_matrix`` で置換する。
    """
    pairs = np.asarray(pair_matrix, dtype=np.int64)
    reference = round_vertices_verma(reference_vertices)

    if pairs.shape != (len(reference), 9):
        raise ValueError(f"pair_matrix must have shape ({len(reference)}, 9).")
    if np.any(pairs < 0) or np.any(pairs > 99):
        raise ValueError("Every two-digit value must be in [0, 99].")

    axis_pairs = pairs.reshape(len(reference), 3, 3)
    mantissa = (
        axis_pairs[:, :, 0] * 10_000
        + axis_pairs[:, :, 1] * 100
        + axis_pairs[:, :, 2]
    )

    scaled_abs = np.rint(np.abs(reference) * _SCALE).astype(np.int64)
    integer_part = scaled_abs // _SCALE
    sign = np.where(np.signbit(reference), -1.0, 1.0)
    reconstructed = sign * (integer_part + mantissa / _SCALE)

    # +0.0 / -0.0 の差は形状に影響しないため、0 は通常の +0.0 に統一する。
    reconstructed[np.isclose(reconstructed, 0.0, atol=0.0)] = 0.0
    return reconstructed.astype(np.float64)


def compute_two_digit_histogram_verma(pair_matrix) -> np.ndarray:
    """00-99 の 100 bin ヒストグラムを返す。"""
    pairs = np.asarray(pair_matrix, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 9:
        raise ValueError("pair_matrix must have shape (N, 9).")
    if np.any(pairs < 0) or np.any(pairs > 99):
        raise ValueError("pair_matrix contains a value outside [0, 99].")
    return np.bincount(pairs.ravel(), minlength=100)[:100]


def find_peak_bin_verma(pair_matrix) -> Tuple[int, np.ndarray]:
    """最頻の 2 桁値とヒストグラムを返す。

    同率 peak の規則は論文にないため、``np.argmax`` により最小 bin を選ぶ。
    """
    histogram = compute_two_digit_histogram_verma(pair_matrix)
    peak_bin = int(np.argmax(histogram))
    return peak_bin, histogram


def find_embedding_positions_verma(pair_matrix, peak_bin: Optional[int] = None):
    """元の M における peak bin の位置を行優先順で返す。

    返り値は shape (K, 2) で、各行は ``[vertex_index, matrix_column]``。
    """
    pairs = np.asarray(pair_matrix, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 9:
        raise ValueError("pair_matrix must have shape (N, 9).")
    if peak_bin is None:
        peak_bin, _ = find_peak_bin_verma(pairs)
    if not 0 <= int(peak_bin) <= 99:
        raise ValueError("peak_bin must be in [0, 99].")
    flat_indices = np.flatnonzero(pairs.ravel(order="C") == int(peak_bin))
    return np.column_stack(np.unravel_index(flat_indices, pairs.shape)).astype(np.int64)


def compute_embedding_capacity_verma(vertices) -> Dict[str, float]:
    """論文方式の最大容量を返す。

    peak 位置 1 個に 3-bit symbol を 2 個、すなわち 6 bit を格納する。
    """
    pair_matrix = vertices_to_pair_matrix_verma(vertices)
    peak_bin, histogram = find_peak_bin_verma(pair_matrix)
    positions = int(histogram[peak_bin])
    return {
        "peak_bin": peak_bin,
        "peak_positions": positions,
        "capacity_bits": positions * _BITS_PER_CELL,
        "bits_per_vertex": (positions * _BITS_PER_CELL) / len(pair_matrix),
    }


def secret_bits_to_payload_values_verma(secret_bits):
    """秘密ビット列を Fig. 4 の 2 桁置換値へ変換する。

    3 bit ごとに 0-7 の数字へ変換し、その数字を 2 個ずつ十の位・一の位へ
    配置する。例: ``011 001 100 101`` -> ``3,1,4,5`` -> ``31,45``。

    ビット数が 6 の倍数でない場合は末尾に 0 を padding する。
    """
    bits = _validate_bits(secret_bits)
    if len(bits) == 0:
        return np.empty(0, dtype=np.int64), 0

    padding = (-len(bits)) % _BITS_PER_CELL
    padded = np.pad(bits, (0, padding), constant_values=0)
    symbols = padded.reshape(-1, _SYMBOL_BITS).dot(
        (1 << np.arange(_SYMBOL_BITS - 1, -1, -1)).astype(np.uint8)
    )
    symbols = symbols.astype(np.int64)
    values = symbols.reshape(-1, 2)[:, 0] * 10 + symbols.reshape(-1, 2)[:, 1]
    return values.astype(np.int64), int(padding)


def payload_values_to_secret_bits_verma(payload_values, secret_bit_length: Optional[int] = None):
    """2 桁置換値から秘密ビット列を復元する。"""
    values = np.asarray(payload_values, dtype=np.int64).reshape(-1)
    if np.any(values < 0) or np.any(values > 77):
        raise ValueError("Payload value must be in 00-77 for two concatenated 3-bit symbols.")

    tens = values // 10
    ones = values % 10
    if np.any(tens >= _SYMBOL_BASE) or np.any(ones >= _SYMBOL_BASE):
        raise ValueError("Each decimal digit of a payload value must be in 0-7.")

    symbols = np.column_stack((tens, ones)).ravel()
    bit_weights = (1 << np.arange(_SYMBOL_BITS - 1, -1, -1)).astype(np.int64)
    bits = ((symbols[:, None] & bit_weights[None, :]) != 0).astype(np.uint8).ravel()

    if secret_bit_length is not None:
        if secret_bit_length < 0 or secret_bit_length > len(bits):
            raise ValueError("secret_bit_length is outside the decodable range.")
        bits = bits[:secret_bit_length]
    return bits


def embed_watermark_verma_mesh(vertices, triangles, secret_bits):
    """Verma らの非ブラインド 3D mesh data hiding を実行する。

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        元メッシュの頂点。
    triangles : array-like, shape (M, 3)
        面接続。方式自体は座標だけを書き換えるが、入力メッシュ検証と main 側の
        一貫性のため受け取る。
    secret_bits : array-like
        0/1 の秘密ビット列。

    Returns
    -------
    watermarked_vertices : ndarray, shape (N, 3)
        小数第 6 位へ正規化し、peak 位置へ秘密値を埋め込んだ頂点。
    key_info : VermaWatermarkKey
        秘密長などの抽出補助情報。抽出位置は元メッシュから再計算する。
    details : dict
        peak、容量、padding、埋め込み位置などの確認情報。
    """
    vertices, triangles = _validate_mesh(vertices, triangles)
    bits = _validate_bits(secret_bits)

    original_pairs = vertices_to_pair_matrix_verma(vertices)
    peak_bin, histogram = find_peak_bin_verma(original_pairs)
    positions = find_embedding_positions_verma(original_pairs, peak_bin)
    payload_values, padding_bits = secret_bits_to_payload_values_verma(bits)

    if len(payload_values) > len(positions):
        raise ValueError(
            "Secret data exceeds the Verma embedding capacity: "
            f"required {len(payload_values)} positions ({len(bits)} bits), "
            f"available {len(positions)} positions ({len(positions) * _BITS_PER_CELL} bits)."
        )

    marked_pairs = original_pairs.copy()
    used_positions = positions[: len(payload_values)]
    if len(used_positions):
        marked_pairs[used_positions[:, 0], used_positions[:, 1]] = payload_values

    watermarked_vertices = pair_matrix_to_vertices_verma(marked_pairs, vertices)
    capacity_bits = len(positions) * _BITS_PER_CELL
    key_info = VermaWatermarkKey(
        secret_bit_length=len(bits),
        used_position_count=len(payload_values),
        peak_bin=peak_bin,
        capacity_bits=capacity_bits,
    )

    details = {
        "peak_bin": peak_bin,
        "histogram": histogram,
        "all_embedding_positions": positions,
        "used_embedding_positions": used_positions,
        "payload_values": payload_values,
        "padding_bits": padding_bits,
        "capacity_bits": capacity_bits,
        "payload_bits": len(bits),
        "embedding_rate_bpv": len(bits) / len(vertices),
        "maximum_embedding_rate_bpv": capacity_bits / len(vertices),
        "rounded_cover_vertices": round_vertices_verma(vertices),
        "original_pair_matrix": original_pairs,
        "marked_pair_matrix": marked_pairs,
    }
    return watermarked_vertices, key_info, details


def extract_watermark_verma_mesh(
    original_vertices,
    marked_vertices,
    triangles,
    key_info: Optional[VermaWatermarkKey] = None,
    secret_bit_length: Optional[int] = None,
    return_details: bool = False,
):
    """元メッシュを用いて、Verma らの秘密ビット列を非ブラインド抽出する。

    基本実装は Fig. 5 に従い、元メッシュの peak 位置を同じ頂点・列番号で
    marked mesh から読み取る。したがって頂点順序と座標系が一致している必要がある。

    ``key_info`` を与えない場合は ``secret_bit_length`` を必ず指定する。
    Hu 版と同様、既定では抽出ビット列だけを返す。確認情報も必要な場合は
    ``return_details=True`` を指定する。
    """
    original_vertices, triangles = _validate_mesh(original_vertices, triangles)
    marked_vertices = np.asarray(marked_vertices, dtype=np.float64)
    if marked_vertices.shape != original_vertices.shape:
        raise ValueError("original_vertices and marked_vertices must have the same shape.")
    if not np.isfinite(marked_vertices).all():
        raise ValueError("marked_vertices contains NaN or infinity.")

    if key_info is not None and not isinstance(key_info, VermaWatermarkKey):
        raise TypeError("key_info must be VermaWatermarkKey.")
    if key_info is None and secret_bit_length is None:
        raise ValueError("Provide key_info or secret_bit_length.")

    original_pairs = vertices_to_pair_matrix_verma(original_vertices)
    marked_pairs = vertices_to_pair_matrix_verma(marked_vertices)
    peak_bin, histogram = find_peak_bin_verma(original_pairs)

    if key_info is not None:
        if key_info.decimal_places != _DECIMAL_PLACES:
            raise ValueError("The key uses an unsupported decimal precision.")
        if peak_bin != key_info.peak_bin:
            raise ValueError(
                f"Peak bin mismatch: original mesh gives {peak_bin}, key expects {key_info.peak_bin}."
            )
        bit_length = key_info.secret_bit_length
        used_position_count = key_info.used_position_count
    else:
        bit_length = int(secret_bit_length)
        used_position_count = int(np.ceil(bit_length / _BITS_PER_CELL))

    positions = find_embedding_positions_verma(original_pairs, peak_bin)
    if used_position_count > len(positions):
        raise ValueError("The original mesh does not contain enough peak positions for this key.")

    used_positions = positions[:used_position_count]
    values = marked_pairs[used_positions[:, 0], used_positions[:, 1]]
    bits = payload_values_to_secret_bits_verma(values, bit_length)

    details = {
        "peak_bin": peak_bin,
        "histogram": histogram,
        "used_embedding_positions": used_positions,
        "extracted_payload_values": values,
        "extracted_bit_length": len(bits),
    }
    if return_details:
        return bits, details
    return bits


def generate_prng_bits_verma(length: int, seed: Optional[int] = None) -> np.ndarray:
    """論文 Fig. 1 の PRNG bits を再現実験用に生成する補助関数。

    論文は PRNG の種類を規定していないため、これは NumPy PCG64 を使う実験用
    ユーティリティであり、提案法固有の必須処理ではない。
    """
    if length < 0:
        raise ValueError("length must be non-negative.")
    return np.random.default_rng(seed).integers(0, 2, size=length, dtype=np.uint8)


