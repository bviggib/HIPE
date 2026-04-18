
COLORS = {
    # fig2_qpstd change start
    "fb_sobol": "dodgerblue",
    "fb_random": "purple",
    "nipv": "forestgreen",
    "qPSTD": "teal",
    "qpstd_iter": "cadetblue",
    "seq_PSTD_BALD": "steelblue",
    "seq_pstdhipe11": "indianred",
    "seq_pstdhipe31": "sienna",
    "hipe": "deeppink",
    "lhsbeta": "darkgoldenrod",
    "bald": "darkgoldenrod",
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
