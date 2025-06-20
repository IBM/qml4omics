### Imports ###

import qbiocode.data_generation.make_circles as circles
import qbiocode.data_generation.make_moons as moons
import qbiocode.data_generation.make_class as make_class
import qbiocode.data_generation.make_s_curve as s_curve
import qbiocode.data_generation.make_spheres as spheres
import qbiocode.data_generation.make_spirals as spirals
import qbiocode.data_generation.make_swiss_roll as swiss_roll

### Main Function ###

# parameters to vary across the configurations
N_SAMPLES = list(range(100, 300, 20))
NOISE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HOLE = [True, False]
N_CLASSES = [2]
DIM = [3, 6, 9, 12]
RAD = [3, 6, 9, 12]
N_FEATURES = list(range(10, 60, 20))
N_INFORMATIVE = list(range(2, 8, 4))
N_REDUNDANT = list(range(2, 8, 4))
N_CLUSTERS_PER_CLASS = list(range(1, 2, 3))
WEIGHTS = [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]]


def generate_data(
        type_of_data=None,
        save_path=None,
        n_samples=N_SAMPLES,
        noise=NOISE,
        hole=HOLE,
        n_classes=N_CLASSES,
        dim=DIM,
        rad=RAD,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_clusters_per_class=N_CLUSTERS_PER_CLASS,
        weights=WEIGHTS
):
    """
    Main function to generate datasets using various methods.
    """

    if type_of_data is 'circles':
        # Generate circles dataset
        circles.my_make_classification(n_samples=n_samples, 
                                       noise=noise, 
                                       save_path=save_path)
    elif type_of_data is 'moons':
        # Generate moons dataset
        moons.my_make_classification(n_samples=n_samples, 
                                     noise=noise,
                                     save_path=save_path)
    elif type_of_data is 'classes':
        # Generate higher-dimensional classification dataset
        make_class.my_make_classification(n_samples=n_samples,
                                            n_features=n_features,
                                            n_informative=n_informative,
                                            n_redundant=n_redundant,
                                            n_classes=n_classes,
                                            n_clusters_per_class=n_clusters_per_class,
                                            weights=weights,
                                            save_path=save_path
        )
    elif type_of_data is 's_curve':
        # Generate S-curve dataset
        s_curve.my_make_s_curve(n_samples=n_samples,
                                noise=noise,
                                save_path=save_path
                                )
    elif type_of_data is 'spheres':
        # Generate spheres dataset
        spheres.my_make_spheres(n_s=n_samples,
                                dim=dim,
                                radius=rad,
                                save_path=save_path
                                )
    elif type_of_data is 'spirals':
        # Generate spirals dataset
        spirals.my_make_spirals(n_s=n_samples,
                                n_c=n_classes,
                                n_n=noise,
                                n_d=dim,
                                save_path=save_path
                                )
    elif type_of_data is 'swiss_roll':
        # Generate Swiss roll dataset
        swiss_roll.my_make_swiss_roll(n_samples=n_samples,
                                    noise=noise,
                                    hole=hole,
                                    save_path=save_path
                                    )
    else:
        raise ValueError("Invalid type_of_data. Choose from 'circles', 'moons', 'classes', 's_curve', 'spheres', 'spirals', or 'swiss_roll'.")

    print("Dataset generation complete.")
    return

