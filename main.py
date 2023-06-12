import numpy as np
from utils.DataLoader import *
from utils.Transformations import *
from utils.menu import *
from models.models import *

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    batch_size = 12
    num_workers = 4
    data_path = "./Dog-Breed-classif"
    dataloaders = dogs_dataset_dataloders(data_transforms_complete, data_path,
                                          batch_size, num_workers)
    returned_objects = menu(dataloaders)
