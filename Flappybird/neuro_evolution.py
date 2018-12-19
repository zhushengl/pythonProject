import random
import math
import copy


def random_weight():
    return random.random() * 2 - 1


def sigmoid(x):
    if x > 50:
        return 1.0
    elif x < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# 神经元类（神经网络的基本单元）
class Neuro(object):
    def __init__(self):
        self.value = 0
        self.weights = []

    def init_weight(self, weight_count):
        for i in range(weight_count):
            self.weights.append(random_weight())


# 神经层类（由神经元组成）
class Layer(object):
    def __init__(self):
        self.neuros = []

    def init_neuros(self, neuros_count, pre_layer_neuros_count):
        for i in range(neuros_count):
            neuro = Neuro()
            neuro.init_weight(pre_layer_neuros_count)
            self.neuros.append(neuro)


# 神经网络（由多个神经层组合而成，必然有输入层和输出层）
class NeuroNetwork(object):
    def __init__(self):
        self.layers = []

    # 所有的神经网络都是由输入层，隐藏层和输出层3个部分组成
    # input -> 输入层神经元个数
    # hiddens -> 隐藏层的层数和各层的神经元数量（[15,12,16,85]）
    # output -> 输出成神经元个数
    def init_neuro_network(self, input, hiddens, output):
        pre_layer_neuros_count = 0
        # 1.初始化输入层
        input_layer = Layer()
        input_layer.init_neuros(input, pre_layer_neuros_count)
        self.layers.append(input_layer)
        pre_layer_neuros_count = len(input_layer.neuros)
        # 2.初始化隐藏层
        for num in hiddens:
            hidden_layer = Layer()
            hidden_layer.init_neuros(num, pre_layer_neuros_count)
            self.layers.append(hidden_layer)
            pre_layer_neuros_count = len(hidden_layer.neuros)
        # 3.初始化输出层
        ouput_layer = Layer()
        ouput_layer.init_neuros(output, pre_layer_neuros_count)
        self.layers.append(ouput_layer)

    def get_data(self):
        # 记录2个值，神经层的结构，神经元的权值
        data = {'layers': [], 'weights': []}
        for layer in self.layers:
            data['layers'].append(len(layer.neuros))
            for n in layer.neuros:
                for weight in n.weights:
                    data['weights'].append(weight)
        return data

    def set_data(self, data):
        self.layers = []
        pre_layer_neuros_count = 0
        index = 0
        for neuro_count in data['layers']:
            layer = Layer()
            layer.init_neuros(neuro_count, pre_layer_neuros_count)
            for neuro in layer.neuros:
                for i in range(len(neuro.weights)):
                    neuro.weights[i] = data['weights'][index]
                    index += 1
            self.layers.append(layer)
            pre_layer_neuros_count = neuro_count

    # 输入一些数据，通过神经网络计算出结果
    def feed_value(self, inputs):
        # inputs的数量必须和输入层的神经元数量一致
        for i in range(len(inputs)):
            self.layers[0].neuros[i].value = inputs[i]

        pre_layer = self.layers[0]
        for layer in self.layers:
            if layer is self.layers[0]:
                continue
            for neuro in layer.neuros:
                sum = 0
                for k in range(len(pre_layer.neuros)):
                    sum += pre_layer.neuros[k].value * neuro.weights[k]
                neuro.value = sigmoid(sum)
            pre_layer = layer
        result = []
        for neuro in self.layers[-1].neuros:
            result.append(neuro.value)
        return result


# 一个个体的基因
class Genome(object):
    def __init__(self, data, score):
        self.data = data  # 一个个体的所有特征值
        self.score = score  # 用来判断个体是否优异


# 世代类（包含多个个体基因）
class Generation(object):
    def __init__(self):
        self.genomes = []

    # 插入一个基因
    def insert_genome(self, genome):
        append_last = True
        for i in range(len(self.genomes)):
            if genome.score > self.genomes[i].score:
                self.genomes.insert(i, genome)
                append_last = False
                break
        if append_last:
            self.genomes.append(genome)

    # 生成下一个世代
    def create_next_data_list(self):
        next = []
        # 1.选取一定比例的精英完全遗传下一代
        for i in range(round(population * elite_ratio)):
            next.append(self.genomes[i].data)
        # 2.选取一定比例的随机行为个体（第一代全部都是随机行为）
        for i in range(round(population * newborn)):
            new_network = NeuroNetwork()
            new_network.init_neuro_network(network[0], network[1], network[2])
            new_date = new_network.get_data()
            next.append(new_date)
        # 3.选取当代的2个个体进行繁衍(剩余的全部由此操作生成)
        while True:
            father = self.genomes[random.randint(0, len(self.genomes) - 1)]
            mother = self.genomes[random.randint(0, len(self.genomes) - 1)]
            if father is mother:
                continue
            child = self.breed(father, mother)
            next.append(child.data)
            if len(next) >= population:
                break
        # 4.将上诉3个方式创建的data打包作为下一代的基因数据
        return next

    def breed(self, father, mother):
        child = copy.deepcopy(father)
        # 交叉
        for i in range(len(child.data['weights'])):
            if random.random() < 0.5:
                child.data['weights'][i] = mother.data['weights'][i]
        # 变异
        for i in range(len(child.data['weights'])):
            if random.random() < mutation_ratio:
                child.data['weights'][i] = random_weight()
        return child


# 每次由个体被淘汰了，将其加入到当代对象中
# 当代对象数量到达种群数量
# 使用当代数据生成下一代，反复执行
class GenerationManager(object):
    def __init__(self):
        self.generations = []

    # 生成第一代,返回data数据列表
    def first_generation(self):
        data_list = []
        for i in range(population):
            new_network = NeuroNetwork()
            new_network.init_neuro_network(network[0], network[1], network[2])
            new_date = new_network.get_data()
            data_list.append(new_date)
        self.generations.append(Generation())
        return data_list

    # 生成下一代,返回data数据列表
    def next_generation(self):
        data_list = self.generations[-1].create_next_data_list()
        self.generations.append(Generation())
        return data_list

    # 添加一个基因组（在最后一代，当前正在适应环境的中添加）
    def add_genome(self, genome):
        if len(self.generations) > 0:
            self.generations[-1].insert_genome(genome)


# 外部使用的类
class AI(object):
    def __init__(self):
        self.manager = GenerationManager()

    def next_generation_network_list(self):
        if len(self.manager.generations) == 0:
            data_list = self.manager.first_generation()
        else:
            data_list = self.manager.next_generation()
        network_list = []
        for data in data_list:
            network = NeuroNetwork()
            network.set_data(data)
            network_list.append(network)
        if historic > 0:  # 至少要保存一代，用于繁殖
            # 当前种群世代列表超过保存值
            if len(self.manager.generations) > historic:
                # 除去historic数量的记录，之前的全部删除，避免占用太多内存
                del self.manager.generations[0:len(self.manager.generations) - historic]
        return network_list

    # 采集分数，增加新的基因
    def gather_score(self, neuro_network, score):
        self.manager.add_genome(Genome(neuro_network.get_data(), score))


network = [3, [7], 1]
population = 100  # 种群个体的数量
elite_ratio = 0.1  # 遗传精英的比例
newborn = 0.1  # 每一代新生个体的比例
mutation_ratio = 0.1  # 产生突变的比率
historic = 1  # 最多保存多少代数据