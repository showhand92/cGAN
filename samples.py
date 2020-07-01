import numpy as np
import torch
import pandas as pd

import helper
from models import Generator

vector_size = 10
batch_size = 100


def generate_excel(selected_epoch, rel_area):
    generator = Generator(vector_size)
    generator.load_state_dict(torch.load('{}/generator_epoch_{}.pth'.format("./out", selected_epoch)))

    noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size, 1)))
    rel_area_input = torch.FloatTensor(np.ones((batch_size, 1)) * rel_area)

    dataset = helper.DisVectorData('./GAN-data-10.xlsx')
    max_v, max_a = dataset.get_max_value()
    real_area = rel_area * max_a

    generated_vector = generator(noise, rel_area_input)
    res = pd.DataFrame(max_v * generated_vector.detach().numpy())

    path = "./results/epoch_{}_rel_{}_area_{:.4f}.xlsx".format(selected_epoch, rel_area, real_area)
    res.to_excel(path)


if __name__ == "__main__":
    for rel_area in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for selected_epoch in [200, 500, 1000, 1500]:
        #for selected_epoch in [200, 500]:
            print("\nGenerating data with epoch: {}, rel area: {}".format(selected_epoch, rel_area))
            generate_excel(selected_epoch, rel_area)

