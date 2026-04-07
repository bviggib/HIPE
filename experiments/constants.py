
COLORS = {
    # fig2_qpstd change start
    "fb_sobol": "dodgerblue",
    "fb_random": "purple",
    "nipv": "forestgreen",
    "qPSTD": "teal",
    "seq_PSTD_BALD": "steelblue",
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
    "seq_PSTD_BALD": "Seq qPSTD->BALD",
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
    "seq_PSTD_BALD",
    "nipv",
    "bald",
    "hipe",
    # fig2_qpstd change end
)
