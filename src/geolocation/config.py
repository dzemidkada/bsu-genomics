AVAILABLE_LOCI = [
    'CSF1PO-0', 'CSF1PO-1', 'D13S317-0', 'D13S317-1', 'D16S539-0', 'D16S539-1',
    'D18S51-0', 'D18S51-1', 'D19S433-0', 'D19S433-1', 'D21S11-0', 'D21S11-1',
    'D2S1338-0', 'D2S1338-1', 'D3S1358-0', 'D3S1358-1', 'D5S818-0', 'D5S818-1',
    'D7S820-0', 'D7S820-1', 'D8S1179-0', 'D8S1179-1', 'FGA-0', 'FGA-1',
    'PentaD-0', 'PentaD-1', 'PentaE-0', 'PentaE-1', 'TH01-0', 'TH01-1',
    'TPOX-0', 'TPOX-1', 'vWA-0', 'vWA-1'
]

DATASETS = {
    'train': 'data/apps/geolocation/train_data.xlsx',
    'val': 'data/apps/geolocation/val_data.xlsx',
    'test': 'data/apps/geolocation/test_data.xlsx'
}

TARGET_AREA = ((51.5, 56), (23.5, 32.5))

HYPERPARAMS_DICT = {
    (0.1, 1): {
        'grid_size': 8.49748071277717,
        'distance_sensitivity': 0.9242445095720511,
        'distance_threshold': 0.9989371871075189,
        'similarity_threshold': 0.40486569441052445,
    },
    (0.1, 5): {
        'grid_size': 8.366524973102319,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.1, 10): {
        'grid_size': 15.638314308394738,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.1, 30): {
        'grid_size': 9.662930893289824,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.3, 1): {
        'grid_size': 7.15276717834931,
        'distance_sensitivity': 0.9165706358414651,
        'distance_threshold': 0.9966827707257724,
        'similarity_threshold': 0.19224766240446378,
    },
    (0.3, 5): {
        'grid_size': 6.188966462922259,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.3, 10): {
        'grid_size': 6.14894898930995,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.3, 30): {
        'grid_size': 9.018144068821982,
        'distance_sensitivity': 0.29864212623925207,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.5, 1): {
        'grid_size': 7.15276717834931,
        'distance_sensitivity': 0.9165706358414651,
        'distance_threshold': 0.9966827707257724,
        'similarity_threshold': 0.19224766240446378,
    },
    (0.5, 5): {
        'grid_size': 8.944968006671695,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.5, 10): {
        'grid_size': 9.039619492953815,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (0.5, 30): {
        'grid_size': 10.0,
        'distance_sensitivity': 0.001,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (1.0, 1): {
        'grid_size': 5.0,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (1.0, 5): {
        'grid_size': 9.434514638349498,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (1.0, 10): {
        'grid_size': 6.150762255931427,
        'distance_sensitivity': 1.0,
        'distance_threshold': 1.0,
        'similarity_threshold': 0.0,
    },
    (1.0, 30): {
        'grid_size': 7.479007735971543,
        'distance_sensitivity': 0.2101830192951897,
        'distance_threshold': 0.7759591443231628,
        'similarity_threshold': 0.09880909744250538,
    }
}
