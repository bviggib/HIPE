
COLORS = {
    # fig2_qpstd change start
    "fb_sobol": "#4E79A7",  # blue
    "fb_random": "#BAB0AC",  # gray
    "nipv": "#59A14F",  # green
    "qPSTD": "#76B7B2",  # teal
    "qpstd_iter": "#EDC948",  # yellow
    "seq_PSTD_BALD": "#9C755F",  # brown
    "seq_pstdhipe11": "#B07AA1",  # purple (highlight)
    "seq_pstdhipe31": "#FF9DA7",  # pink
    "hipe": "#E15759",  # red (highlight)
    "lhsbeta": "#4E79A7",  # blue
    "bald": "#F28E2B",  # orange (highlight)
    # fig2_qpstd change end
}

NAMES = {
    # fig2_qpstd change start
    "hipe": "HIPE",
    "sobol": "Sobol",
    "random": "Random",
    "lhsbeta": "LHS-Beta",
    "nipv": "NIPV",
    "bald": "BALD",
    "qPSTD": "qPSTD",
    "qpstd_iter": "qPSTD-Iter",
    "seq_PSTD_BALD": "Seq qPSTD->BALD",
    "seq_pstdhipe11": "Seq qPSTD->HIPE (2/2)",
    "seq_pstdhipe31": "Seq qPSTD->HIPE (3/1)",
    # fig2_qpstd change end
}

BENCHMARKS = {
    "hartmann6": "Hartmann $6D$",
    "hartmann6_12": "Hartmann $6D_e$ ($12D$)",
    "hartmann4": "Hartmann $4D$",
    "hartmann4_8": "Hartmann $4D_e$ ($8D$)",
    "ackley4": "Ackley $4D_e$",
    "Fashion-MNIST": "Fashion-MNIST",
    "higgs": "Higgs",
    "segment": "Segment",
    "MiniBooNE": "MiniBooNE",
    "car": "Car",
    "Australian": "Australian",
    "ishigami": "Ishigami",
    "svm_20": "SVM $20D$",
    "svm_40": "SVM $40D$",
}

METRIC_NAMES = {
    "MLL": "Negative Log-Likelihood",
    "RMSE": "Root Mean Squared Error",
}

BO_METHOD_ORDER = (
    "random",
    "sobol",
    "lhsbeta",
    "nipv",
    "hipe",
)
AL_METHOD_ORDER = (
    # fig2_qpstd change start
    "random",
    "sobol",
    "qPSTD",
    "qpstd_iter",
    "seq_PSTD_BALD",
    "seq_pstdhipe11",
    "seq_pstdhipe31",
    "nipv",
    "bald",
    "hipe",
    # fig2_qpstd change end
)
