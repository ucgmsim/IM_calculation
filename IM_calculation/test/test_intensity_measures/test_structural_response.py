import numpy as np
import pytest

from IM_calculation.IM.structural_response import (
    sa_sd_time,
    calculate_vibration_periods,
    calculate_mode_shapes,
    calculate_structural_response_B,
    calculate_structural_response,
    extract_peak_structural_response,
)

BENCH_VIBRATION_PERIOD_02 = np.asarray(
    [
        0.5,
        0.07978432,
        0.02849412,
        0.01454078,
        0.00879622,
        0.00588838,
        0.00421594,
        0.00316664,
    ]
)
BENCH_VIBRATION_PERIOD_01 = np.asarray(
    [2.0, 0.65996249, 0.38830765, 0.26974194, 0.20263475, 0.15926433, 0.12893242]
)

T1_01 = 2.0
T1_02 = 0.5

BENCH_PARTICIPATION_FACTOR_01 = np.asarray([1, 1, 1, 1, 1, 1, 1])
BENCH_PARTICIPATION_FACTOR_02 = np.asarray(
    [
        1.20106878,
        0.18971139,
        1.01929249,
        1.13806637,
        1.00085071,
        0.19526212,
        1.00003679,
        1.13807119,
    ]
)

BENCH_PHI_4_01 = np.asarray(
    [
        [
            43863.17124484,
            -131471.94997989,
            218942.95722839,
            -307310.07877287,
            399809.37601853,
            -482181.96518574,
            515448.64192754,
        ],
        [
            2421.21212858,
            21937.767748,
            63100.14867257,
            132361.23278689,
            241490.23030303,
            364722.08695652,
            299593.14285714,
        ],
    ]
)
BENCH_PHI_4_02 = np.asarray(
    [
        [0, -0, 0, -0, 0, -0, 0, -0],
        [
            4.19730820,
            -346.498124,
            74.9416552,
            1.03361465e04,
            34.0096459,
            -6.30282515e04,
            6.39785104,
            2.17936286e05,
        ],
        [
            12.3623637,
            485.519165,
            3806.54639,
            1.46172733e04,
            3.99438318e04,
            8.91354051e04,
            1.73881303e05,
            3.08208500e05,
        ],
    ]
)

BENCH_PHI_3_01 = np.asarray(
    [
        [
            -1464.24817492,
            4439.69715067,
            -7558.86903761,
            10945.11459756,
            -14795.28705585,
            18639.49722225,
            -20892.85342802,
        ],
        [
            80.64070001,
            697.10954396,
            1899.3840708,
            3706.5442623,
            5957.81818182,
            8548.17391304,
            9362.28571429,
        ],
    ]
)
BENCH_PHI_3_02 = np.asarray(
    [
        [
            -4.83981435,
            105.341988,
            -484.240813,
            1329.42655,
            -2825.44374,
            5158.66957,
            -8515.09848,
            13080.7730,
        ],
        [
            -4.08931723,
            9.98471923,
            342.542862,
            5.44509607,
            -1997.89188,
            0.913070993,
            6021.08389,
            0.100042760,
        ],
        [
            7.44208483e-08,
            8.65751529e-05,
            2.99311292e-05,
            -6.48222264e-06,
            -3.21864615e-06,
            -3.81469752e-05,
            4.88281235e-04,
            -0,
        ],
    ]
)

BENCH_PHI_2_01 = np.asarray(
    [
        [48.73686, -146.07994, 243.26995, -341.45564, 444.23264, -535.75774, 572.72071],
        [0, -0, 0, -1.67869, 0, -0, 0],
    ]
)
BENCH_PHI_2_02 = np.asarray(
    [
        [
            3.51601528,
            -22.0344806,
            61.6972116,
            -120.901917,
            199.859530,
            -298.555532,
            416.990775,
            -555.165247,
        ],
        [
            1.19376844,
            15.7252568,
            1.21467105,
            -85.4919993,
            0.170167894,
            211.110648,
            0.0153425997,
            -392.561111,
        ],
        [0, -0, 0, -0, 0, -0, 0.0000152587886, -0],
    ]
)

BENCH_PHI_1_01 = np.asarray(
    [
        [0, -0, 0, -0, 0, -0, 0],
        [0.08944522, 0.78232, 2.09558, 3.93443, 6.4, 9.73913, 18.28571],
    ]
)
BENCH_PHI_1_02 = np.asarray(
    [
        [0, -0, 0, -0, 0, -0, 0, -0],
        [
            1.16305445,
            0.453143084,
            -5.55199873,
            0.0450373397,
            9.99648040,
            0.00305826985,
            -14.4393691,
            0.000180233852,
        ],
        [
            1.37650548,
            4.78077519,
            7.84866562,
            10.9959097,
            14.1371479,
            17.2787606,
            20.4203519,
            23.5619354,
        ],
    ]
)

BENCH_PHI_01 = np.asarray([[0, -0, 0, -0, 0, -0, 0], [1, 1, 1, 1, 1, 1, 1]])
BENCH_PHI_02 = np.asarray(
    [
        [0, -0, 0, -0, 0, -0, 0, -0],
        [
            0.33952311355263276,
            -0.7136651832681893,
            0.019687571779099335,
            0.7071186475888285,
            0.0008514367392038657,
            -0.7071068057916756,
            3.679435466831145e-05,
            0.7071067811448302,
        ],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
)

BENCH_SD_TIME_01 = np.asarray(
    [
        [0, -2.74330120e-09, -1.81055521e-08, -5.86562429e-08, -1.23222546e-07],
        [0, -2.73755843e-09, -1.80534434e-08, -5.84200711e-08, -1.22506395e-07],
        [0, -2.73011351e-09, -1.79837400e-08, -5.80933957e-08, -1.21480164e-07],
        [0, -2.72055658e-09, -1.78924180e-08, -5.76566466e-08, -1.20080998e-07],
        [0, -2.70835534e-09, -1.77741953e-08, -5.70839828e-08, -1.18225914e-07],
        [0, -2.69287070e-09, -1.76227406e-08, -5.63446856e-08, -1.15818034e-07],
        [0, -2.67337469e-09, -1.74309274e-08, -5.54049304e-08, -1.12754675e-07],
    ]
)
BENCH_SD_TIME_02 = np.asarray(
    [
        [0, -8.97814418e-10, -6.01796991e-09, -2.04188008e-08, -4.73698232e-08],
        [0, -7.59362122e-10, -4.66689558e-09, -1.38764949e-08, -2.63099618e-08],
        [0, -3.89964947e-10, -1.77262083e-09, -3.41795496e-09, -3.42938381e-09],
        [0, -1.54147256e-10, -5.37084613e-10, -8.08063154e-10, -7.35223433e-10],
        [0, -6.42672188e-11, -1.96980466e-10, -2.86444872e-10, -2.69819836e-10],
        [0, -3.02350250e-11, -8.76783179e-11, -1.28144767e-10, -1.20220549e-10],
        [0, -1.58364251e-11, -4.47671013e-11, -6.58162111e-11, -6.12354069e-11],
        [0, -9.03238934e-12, -2.52055595e-11, -3.71916716e-11, -3.43989173e-11],
    ]
)

BENCH_SA_TIME_01 = np.asarray(
    [
        [0, -4.38928192e-04, -1.14117556e-03, -1.30887108e-03, -8.35802089e-05],
        [0, -4.38009349e-04, -1.13651354e-03, -1.29708243e-03, -6.44729475e-05],
        [0, -4.36818162e-04, -1.13012575e-03, -1.27989502e-03, -3.44221770e-05],
        [0, -4.35289053e-04, -1.12163066e-03, -1.25622838e-03, 8.46786462e-06],
        [0, -4.33336855e-04, -1.11052383e-03, -1.22464707e-03, 6.66749772e-05],
        [0, -4.30859313e-04, -1.09620125e-03, -1.18347021e-03, 1.42917216e-04],
        [0, -4.27739950e-04, -1.07798859e-03, -1.13091490e-03, 2.39699684e-04],
    ]
)
BENCH_SA_TIME_02 = np.asarray(
    [
        [0, -3.59125767e-05, -9.70684896e-05, -1.41177459e-04, -1.22584256e-04],
        [0, -3.03744849e-05, -6.51778835e-05, -5.13523822e-05, 3.89279444e-05],
        [0, -1.55985979e-05, -8.51044190e-06, 2.21123522e-05, 2.96419485e-05],
        [0, -6.16589024e-06, 3.18017643e-06, 4.28388997e-06, 2.00477416e-06],
        [0, -2.57068875e-06, 2.40353639e-06, -5.06430394e-07, 2.85290208e-06],
        [0, -1.20940100e-06, 1.33047129e-06, -7.72467811e-07, 2.15009100e-06],
        [0, -6.33457003e-07, 7.43143960e-07, -5.37568260e-07, 1.35718912e-06],
        [0, -3.61295574e-07, 4.36959916e-07, -3.45141937e-07, 8.44478617e-07],
    ]
)

ALPHA_01 = 30
ALPHA_02 = 0

STOREY_01 = 1
STOREY_02 = 2

ACC_TIME_HISTORY_01 = np.asarray(
    [0, 0.000044780836, 0.000116507645, 0.000133838940, 0.000009113150]
)
ACC_TIME_HISTORY_02 = np.asarray(
    [0, 0.000003698267, 0.000010099898, 0.000014980632, 0.000013688073]
)

DT_01 = 0.005
DT_02 = 0.01

BENCH_STOREY_MOMENT_01 = [
    [0, -7.55946296e-07, -4.91797363e-06, -1.55761411e-05, -3.15046531e-05],
    [0, 4.56696712e-09, 3.00357967e-08, 9.67875510e-08, 2.01578593e-07],
]
BENCH_STOREY_MOMENT_02 = [
    [0, -2.18093332e-09, -4.53919548e-08, -1.70758613e-07, -3.24083543e-07],
    [0, 1.37401665e-08, 3.51225394e-08, 1.49992202e-08, -6.87546474e-08],
    [0, -2.41653552e-16, -6.83116866e-16, -1.00431260e-15, -9.34412504e-16],
]


BENCH_STOREY_SHEAR_01 = [
    [0, 4.70020114e-05, 3.06038096e-04, 9.70590523e-04, 1.96771491e-03],
    [0, 1.85477368e-05, 1.20899616e-04, 3.84104495e-04, 7.81062120e-04],
]
BENCH_STOREY_SHEAR_02 = [
    [0, -2.03110901e-11, 1.52935271e-09, 4.56282255e-08, 1.32819572e-07],
    [0, -1.00994976e-07, -4.77218345e-07, -9.47942409e-07, -8.48321095e-07],
    [0, -3.06130101e-14, -1.47878432e-13, -3.58313178e-13, -5.63694890e-13],
]

BENCH_GROUND_ACCEL_01 = [
    [0, 4.39300001e-04, 1.14294000e-03, 1.31296000e-03, 8.94000015e-05],
    [0, 4.39300001e-04, 1.14294000e-03, 1.31296000e-03, 8.94000015e-05],
]
BENCH_GROUND_ACCEL_02 = [
    [0, 3.62799993e-05, 9.90799994e-05, 1.46960000e-04, 1.34279996e-04],
    [0, 3.62799993e-05, 9.90799994e-05, 1.46960000e-04, 1.34279996e-04],
    [0, 3.62799993e-05, 9.90799994e-05, 1.46960000e-04, 1.34279996e-04],
]

BENCH_TOTAL_ACCEL_01 = [
    [0, 4.39300001e-04, 1.14294000e-03, 1.31296000e-03, 8.94000015e-05],
    [0, -2.60168087e-03, -6.67121919e-03, -7.36814908e-03, 3.64684410e-04],
]
BENCH_TOTAL_ACCEL_02 = [
    [0, 3.62799993e-05, 9.90799994e-05, 1.46960000e-04, 1.34279996e-04],
    [0, 2.03466122e-05, 7.08793133e-05, 1.00061422e-04, 8.16141907e-05],
    [0, -3.93862734e-05, -3.10204630e-05, -6.51974337e-06, 3.25218052e-05],
]

BENCH_GAMMA_01 = np.asarray(
    [
        1.62466173,
        4.86707584,
        8.09106737,
        11.29056376,
        14.4651604,
        17.61837538,
        20.75549639,
    ]
)
BENCH_GAMMA_02 = np.asarray(
    [
        1.87510408,
        4.69409197,
        7.8547575,
        10.99554073,
        14.13716839,
        17.27875953,
        20.42035225,
        23.5619449,
    ]
)

BENCH_DISP_SLOPE_01 = [
    [0, 0, 0, 0, 0],
    [0, -1.90061305e-08, -1.24863017e-07, -4.01659955e-07, -8.34088724e-07],
]
BENCH_DISP_02 = [
    [0, 0, 0, 0, 0],
    [0, -3.98334677e-10, -2.29835586e-09, -7.17921046e-09, -1.64266562e-08],
    [0, -1.89165682e-09, -1.08191388e-08, -3.19802944e-08, -6.66119034e-08],
]
BENCH_SLOPE_02 = [
    [0, 0, 0, 0, 0],
    [0, -7.96669353e-10, -4.59671172e-09, -1.43584209e-08, -3.28533124e-08],
    [0, -2.98664428e-09, -1.70415659e-08, -4.96021678e-08, -1.00370494e-07],
]


BENCH_MOMENT_01 = [
    [0, -7.60513263e-07, -4.94800943e-06, -1.56729286e-05, -3.17062317e-05],
    [0, 4.56696712e-09, 3.00357967e-08, 9.67875510e-08, 2.01578593e-07],
]
BENCH_MOMENT_02 = [
    [0, -1.59210998e-08, -8.05144941e-08, -1.85757833e-07, -2.55328896e-07],
    [0, 1.37401667e-08, 3.51225401e-08, 1.49992212e-08, -6.87546465e-08],
    [0, -2.41653552e-16, -6.83116866e-16, -1.00431260e-15, -9.34412504e-16],
]

BENCH_SHEAR_01 = [
    [0, 2.84542746e-05, 1.85138480e-04, 5.86486028e-04, 1.18665279e-03],
    [0, 1.85477368e-05, 1.20899616e-04, 3.84104495e-04, 7.81062120e-04],
]
BENCH_SHEAR_02 = [
    [0, 1.00974665e-07, 4.78747698e-07, 9.93570635e-07, 9.81140667e-07],
    [0, -1.00994946e-07, -4.77218197e-07, -9.47942051e-07, -8.48320531e-07],
    [0, -3.06130101e-14, -1.47878432e-13, -3.58313178e-13, -5.63694890e-13],
]

BENCH_LOAD_01 = [
    [0, 9.84411306e-12, 6.55643901e-11, 2.15395293e-10, 4.62733075e-10],
    [0, -3.04029060e-03, -1.99118446e-02, -6.37422535e-02, -1.31299097e-01],
]
BENCH_LOAD_02 = [
    [0, 0, 0, 0, 0],
    [0, -3.66812838e-06, -1.13564035e-05, -1.66149130e-05, -1.44817637e-05],
    [0, -1.31781236e-05, -4.23580889e-05, -6.64572787e-05, -6.42566081e-05],
]

BENCH_REL_ACCEL_01 = [
    [0, 0, 0, 0, 0],
    [0, -0.00304098, -0.0078141, -0.0086811, 0.000275284],
]
BENCH_REL_ACCEL_02 = [
    [0, 0, 0, 0, 0],
    [0, -1.59333871e-05, -2.82006861e-05, -4.68985776e-05, -5.26658054e-05],
    [0, -7.56662727e-05, -1.30100462e-04, -1.53479743e-04, -1.01758191e-04],
]


def compare_is_close(test_vals, bench_vals):
    return np.isclose(test_vals, bench_vals, atol=1.0e-20).all()


@pytest.mark.parametrize(
    ["alpha", "t1", "bench_gamma", "bench_frequency"],
    [
        (ALPHA_01, T1_01, BENCH_GAMMA_01, BENCH_VIBRATION_PERIOD_01),
        (ALPHA_02, T1_02, BENCH_GAMMA_02, BENCH_VIBRATION_PERIOD_02),
    ],
)
def test_calculate_vibration_periods(alpha, t1, bench_gamma, bench_frequency):
    test_gamma, test_frequency = calculate_vibration_periods(alpha, t1)
    assert compare_is_close(test_gamma, bench_gamma)
    assert compare_is_close(test_frequency, bench_frequency)


@pytest.mark.parametrize(
    [
        "alpha",
        "gamma",
        "storey",
        "bench_phi",
        "bench_phi_1",
        "bench_phi_2",
        "bench_phi_3",
        "bench_phi_4",
        "bench_participation_factor",
    ],
    [
        (
            ALPHA_01,
            BENCH_GAMMA_01,
            STOREY_01,
            BENCH_PHI_01,
            BENCH_PHI_1_01,
            BENCH_PHI_2_01,
            BENCH_PHI_3_01,
            BENCH_PHI_4_01,
            BENCH_PARTICIPATION_FACTOR_01,
        ),
        (
            ALPHA_02,
            BENCH_GAMMA_02,
            STOREY_02,
            BENCH_PHI_02,
            BENCH_PHI_1_02,
            BENCH_PHI_2_02,
            BENCH_PHI_3_02,
            BENCH_PHI_4_02,
            BENCH_PARTICIPATION_FACTOR_02,
        ),
    ],
)
def test_calculate_mode_shapes(
    alpha,
    gamma,
    storey,
    bench_phi,
    bench_phi_1,
    bench_phi_2,
    bench_phi_3,
    bench_phi_4,
    bench_participation_factor,
):
    test_phi, test_phi_1, test_phi_2, test_phi_3, test_phi_4, test_participation_factor = calculate_mode_shapes(
        alpha, gamma, storey
    )

    assert compare_is_close(test_phi, bench_phi)
    assert compare_is_close(test_phi_1, bench_phi_1)
    assert compare_is_close(test_phi_2, bench_phi_2)
    assert compare_is_close(test_phi_3, bench_phi_3)
    assert compare_is_close(test_phi_4, bench_phi_4)
    assert compare_is_close(test_participation_factor, bench_participation_factor)


@pytest.mark.parametrize(
    ["acc_time_history", "dt", "vibration_period", "bench_sa_time", "bench_sd_time"],
    [
        (
            ACC_TIME_HISTORY_01,
            DT_01,
            BENCH_VIBRATION_PERIOD_01,
            BENCH_SA_TIME_01,
            BENCH_SD_TIME_01,
        ),
        (
            ACC_TIME_HISTORY_02,
            DT_02,
            BENCH_VIBRATION_PERIOD_02,
            BENCH_SA_TIME_02,
            BENCH_SD_TIME_02,
        ),
    ],
)
def test_sa_sd_time(
    acc_time_history, dt, vibration_period, bench_sa_time, bench_sd_time
):
    test_sa_time, test_sd_time = sa_sd_time(
        acc_time_history, dt, t1_range=vibration_period
    )
    assert compare_is_close(test_sa_time, bench_sa_time)
    assert compare_is_close(test_sd_time, bench_sd_time)


@pytest.mark.parametrize(
    [
        "alpha",
        "gamma",
        "storey",
        "acc_time_history",
        "dt",
        "vibration_period",
        "phi",
        "phi_1",
        "phi_2",
        "phi_3",
        "phi_4",
        "participation_factor",
        "bench_disp",
        "bench_slope",
        "bench_moment",
        "bench_storey_moment",
        "bench_shear",
        "bench_storey_shear",
        "bench_load",
        "bench_ground_accel",
        "bench_rel_accel",
        "bench_total_accel",
    ],
    [
        (
            ALPHA_01,
            BENCH_GAMMA_01,
            STOREY_01,
            ACC_TIME_HISTORY_01,
            DT_01,
            BENCH_VIBRATION_PERIOD_01,
            BENCH_PHI_01,
            BENCH_PHI_1_01,
            BENCH_PHI_2_01,
            BENCH_PHI_3_01,
            BENCH_PHI_4_01,
            BENCH_PARTICIPATION_FACTOR_01,
            BENCH_DISP_SLOPE_01,
            BENCH_DISP_SLOPE_01,
            BENCH_MOMENT_01,
            BENCH_STOREY_MOMENT_01,
            BENCH_SHEAR_01,
            BENCH_STOREY_SHEAR_01,
            BENCH_LOAD_01,
            BENCH_GROUND_ACCEL_01,
            BENCH_REL_ACCEL_01,
            BENCH_TOTAL_ACCEL_01,
        ),
        (
            ALPHA_02,
            BENCH_GAMMA_02,
            STOREY_02,
            ACC_TIME_HISTORY_02,
            DT_02,
            BENCH_VIBRATION_PERIOD_02,
            BENCH_PHI_02,
            BENCH_PHI_1_02,
            BENCH_PHI_2_02,
            BENCH_PHI_3_02,
            BENCH_PHI_4_02,
            BENCH_PARTICIPATION_FACTOR_02,
            BENCH_DISP_02,
            BENCH_SLOPE_02,
            BENCH_MOMENT_02,
            BENCH_STOREY_MOMENT_02,
            BENCH_SHEAR_02,
            BENCH_STOREY_SHEAR_02,
            BENCH_LOAD_02,
            BENCH_GROUND_ACCEL_02,
            BENCH_REL_ACCEL_02,
            BENCH_TOTAL_ACCEL_02,
        ),
    ],
)
def test_calculate_structural_response(
    alpha,
    gamma,
    storey,
    acc_time_history,
    dt,
    vibration_period,
    phi,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
    participation_factor,
    bench_disp,
    bench_slope,
    bench_moment,
    bench_storey_moment,
    bench_shear,
    bench_storey_shear,
    bench_load,
    bench_ground_accel,
    bench_rel_accel,
    bench_total_accel,
):
    test_disp, test_slope, test_moment, test_storey_moment, test_shear, test_storey_shear, test_load, test_ground_accel, test_rel_accel, test_total_accel = calculate_structural_response(
        vibration_period,
        acc_time_history,
        dt,
        storey,
        participation_factor,
        phi,
        phi_1,
        phi_2,
        phi_3,
        phi_4,
        alpha,
    )
    assert compare_is_close(test_disp, bench_disp)
    assert compare_is_close(test_slope, bench_slope)
    assert compare_is_close(test_moment, bench_moment)
    assert compare_is_close(test_storey_moment, bench_storey_moment)
    assert compare_is_close(test_shear, bench_shear)
    assert compare_is_close(test_storey_shear, bench_storey_shear)
    assert compare_is_close(test_load, bench_load)
    assert compare_is_close(test_ground_accel, bench_ground_accel)
    assert compare_is_close(test_rel_accel, bench_rel_accel)
    assert compare_is_close(test_total_accel, bench_total_accel)


@pytest.mark.parametrize(
    [
        "alpha",
        "storey",
        "phi",
        "phi_1",
        "phi_2",
        "phi_3",
        "phi_4",
        "participation_factor",
        "sa_time",
        "sd_time",
        "bench_disp",
        "bench_slope",
        "bench_moment",
        "bench_shear",
        "bench_load",
        "bench_rel_accel",
    ],
    [
        (
            ALPHA_01,
            STOREY_01,
            BENCH_PHI_01,
            BENCH_PHI_1_01,
            BENCH_PHI_2_01,
            BENCH_PHI_3_01,
            BENCH_PHI_4_01,
            BENCH_PARTICIPATION_FACTOR_01,
            BENCH_SA_TIME_01,
            BENCH_SD_TIME_01,
            BENCH_DISP_SLOPE_01,
            BENCH_DISP_SLOPE_01,
            BENCH_MOMENT_01,
            BENCH_SHEAR_01,
            BENCH_LOAD_01,
            BENCH_REL_ACCEL_01,
        ),
        (
            ALPHA_02,
            STOREY_02,
            BENCH_PHI_02,
            BENCH_PHI_1_02,
            BENCH_PHI_2_02,
            BENCH_PHI_3_02,
            BENCH_PHI_4_02,
            BENCH_PARTICIPATION_FACTOR_02,
            BENCH_SA_TIME_02,
            BENCH_SD_TIME_02,
            BENCH_DISP_02,
            BENCH_SLOPE_02,
            BENCH_MOMENT_02,
            BENCH_SHEAR_02,
            BENCH_LOAD_02,
            BENCH_REL_ACCEL_02,
        ),
    ],
)
def test_calculate_structural_response_b(
    alpha,
    storey,
    phi,
    phi_1,
    phi_2,
    phi_3,
    phi_4,
    participation_factor,
    sa_time,
    sd_time,
    bench_disp,
    bench_slope,
    bench_moment,
    bench_shear,
    bench_load,
    bench_rel_accel,
):
    test_disp, test_slope, test_moment, test_shear, test_load, test_rel_accel = calculate_structural_response_B(
        participation_factor,
        phi,
        phi_1,
        phi_2,
        phi_3,
        phi_4,
        sa_time,
        sd_time,
        alpha,
        storey,
    )
    assert compare_is_close(test_disp, bench_disp)
    assert compare_is_close(test_slope, bench_slope)
    assert compare_is_close(test_moment, bench_moment)
    assert compare_is_close(test_shear, bench_shear)
    assert compare_is_close(test_load, bench_load)
    assert compare_is_close(test_rel_accel, bench_rel_accel)


@pytest.mark.parametrize(
    [
        "storey",
        "disp",
        "slope",
        "moment",
        "storey_moment",
        "shear",
        "storey_shear",
        "load",
        "ground_accel",
        "rel_accel",
        "total_accel",
        "bench_disp",
        "bench_slope",
        "bench_moment",
        "bench_storey_moment",
        "bench_shear",
        "bench_storey_shear",
        "bench_load",
        "bench_ground_accel",
        "bench_rel_accel",
        "bench_total_accel",
    ],
    [
        (
            STOREY_01,
            BENCH_DISP_SLOPE_01,
            BENCH_DISP_SLOPE_01,
            BENCH_MOMENT_01,
            BENCH_STOREY_MOMENT_01,
            BENCH_SHEAR_01,
            BENCH_STOREY_SHEAR_01,
            BENCH_LOAD_01,
            BENCH_GROUND_ACCEL_01,
            BENCH_REL_ACCEL_01,
            BENCH_TOTAL_ACCEL_01,
            [0, 8.34088724e-07],
            [0, 8.34088724e-07],
            [3.17062317e-05, 0],
            [3.15046531e-05, 0],
            [0.00118665, 0],
            [0.00196771, 0],
            [4.62733080e-10, 1.31299096e-01],
            [0.00131296, 0.00131296],
            [0, 0.00868111],
            [0.00131296, 0.00736815],
        ),
        (
            STOREY_02,
            BENCH_DISP_02,
            BENCH_SLOPE_02,
            BENCH_MOMENT_02,
            BENCH_STOREY_MOMENT_02,
            BENCH_SHEAR_02,
            BENCH_STOREY_SHEAR_02,
            BENCH_LOAD_02,
            BENCH_GROUND_ACCEL_02,
            BENCH_REL_ACCEL_02,
            BENCH_TOTAL_ACCEL_02,
            np.max(np.abs(BENCH_DISP_02), axis=1),
            np.max(np.abs(BENCH_SLOPE_02), axis=1),
            [2.55328896e-07, 6.87546465e-08, 0],
            [3.24083543e-07, 6.87546474e-08, 0],
            [9.93570635e-07, 9.47942051e-07, 0],
            [1.32819572e-07, 9.47942409e-07, 0],
            np.max(np.abs(BENCH_LOAD_02), axis=1),
            np.max(np.abs(BENCH_GROUND_ACCEL_02), axis=1),
            np.max(np.abs(BENCH_REL_ACCEL_02), axis=1),
            np.max(np.abs(BENCH_TOTAL_ACCEL_02), axis=1),
        ),
    ],
)
def test_extract_peak_structural_response(
    storey,
    disp,
    slope,
    moment,
    storey_moment,
    shear,
    storey_shear,
    load,
    ground_accel,
    rel_accel,
    total_accel,
    bench_disp,
    bench_slope,
    bench_moment,
    bench_storey_moment,
    bench_shear,
    bench_storey_shear,
    bench_load,
    bench_ground_accel,
    bench_rel_accel,
    bench_total_accel,
):
    test_disp, test_slope, test_moment, test_storey_moment, test_shear, test_storey_shear, test_load, test_ground_accel, test_rel_accel, test_total_accel = extract_peak_structural_response(
        disp,
        slope,
        moment,
        storey_moment,
        shear,
        storey_shear,
        load,
        ground_accel,
        rel_accel,
        total_accel,
    )
    assert compare_is_close(test_disp, bench_disp)
    assert compare_is_close(test_slope, bench_slope)
    assert compare_is_close(test_moment, bench_moment)
    assert compare_is_close(test_storey_moment, bench_storey_moment)
    assert compare_is_close(test_shear, bench_shear)
    assert compare_is_close(test_storey_shear, bench_storey_shear)
    assert compare_is_close(test_load, bench_load)
    assert compare_is_close(test_ground_accel, bench_ground_accel)
    assert compare_is_close(test_rel_accel, bench_rel_accel)
    assert compare_is_close(test_total_accel, bench_total_accel)
