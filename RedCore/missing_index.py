import numpy as np
import torch



# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子
# setup_seed(20)



def missing_pattern(num_modality, num_sample, ratio):
    missing_matrix = np.ones((num_sample, num_modality))
    for i in range(0, num_modality):
        missing_index = np.random.choice(np.arange(num_sample), replace=False, size=int(num_sample * ratio[i]))
        missing_matrix[missing_index, i] = 0
    
    missing_matrix = torch.tensor(missing_matrix)
    return missing_matrix

if __name__ == '__main__':
    a = missing_pattern(3, 10, [0.2, 0.5, 0.8])
    print(a)






        # self.A_num = len(self.all_A)
        # self.V_num = len(self.all_V)
        # self.L_num = len(self.all_L)
        # self.A_miss_matrix = np.random.random_integers(1, 1, size=self.A_num)
        # self.V_miss_matrix = np.random.random_integers(1, 1, size=self.V_num)
        # self.L_miss_matrix = np.random.random_integers(1, 1, size=self.L_num)
        # # self.A_miss_matrix = [1] * self.A_num
        # # self.V_miss_matrix = [1] * self.V_num
        # # self.L_miss_matrix = [1] * self.L_num
        # miss_indices_a = np.random.choice(np.arange(self.A_miss_matrix.size), replace=False,
        #                            size=int(self.A_miss_matrix.size * opt.A_miss_ratio))
        # miss_indices_v = np.random.choice(np.arange(self.V_miss_matrix.size), replace=False,
        #                            size=int(self.V_miss_matrix.size * opt.V_miss_ratio))
        # miss_indices_l = np.random.choice(np.arange(self.L_miss_matrix.size), replace=False,
        #                            size=int(self.L_miss_matrix.size * opt.L_miss_ratio))

        # self.A_miss_matrix[miss_indices_a] = 0
        # self.V_miss_matrix[miss_indices_v] = 0
        # self.L_miss_matrix[miss_indices_l] = 0

