import math
import random
import copy

# 神经网络3层,1个输入层4个神经元，1个隐藏层16个神经元，1个输出层1个神经元
network = [4, [16], 1]  # 输入和输出都是1层，隐藏层可以有多个，理论上越多，结果越精确
# 遗传算法相关
population = 50  # 每一代的种群数量
elitism = 0.2  # 精英比率，即从种群中选取多少遗传到下一代
random_behaviour = 0.1  # 随机行为的比率，种群*比率数量的个体，随机其行为权重
mutation_rate = 0.3  # 神经元权重值，突变的概率
historic = 1  # 保存历史世代的数量，至少要保存一代，用于生成下一代数据


# sigmoid激活函数,将任意一个实数映射到(0，1)
def sigmoid(z):
    if z > 700:
        return 1.0
    elif z < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


# 随机权重值(-1.0,1.0)
def random_weight():
    return random.random() * 2 - 1


# 神经元类
class Neuron(object):
    def __init__(self):
        self.value = 0  # 数学计算值
        self.weights = []  # 权重列表

    # 初始化权重值，n为权重值的数量，第一代需要随机初始化，后面根据上一代生成
    def init_weights(self, n):
        for i in range(n):
            self.weights.append(random_weight())


# 神经网络层
class Layer(object):
    def __init__(self):
        self.neurons = []  # 每一层当中包含N个神经元

    # 初始化神经元，n_neuron为神经元的数量，n_weight为神经元包含的权重值数量
    # 第一代需要随机初始化，后面根据上一代生成
    def init_neurons(self, n_neuron, n_weight):
        self.neurons = []  # 先清空原有数据
        for i in range(n_neuron):
            neuron = Neuron()
            neuron.init_weights(n_weight)
            self.neurons.append(neuron)


# 神经网络
class NeuroNetwork(object):
    def __init__(self):
        self.layers = []  # 神经网络由多层组成，这里一共三层

    # 初始化神经网络，参数要指定输入层，隐藏层(可能多层)，输出层
    def init_neuro_network(self, input, hiddens, output):
        # 上一层神经元个数，决定了当前层每个神经元的权重值数量
        # 第一层没有权重，数量为0
        previous_neurons = 0

        # input层初始化
        layer = Layer()
        # input神经元数量，previous_neurons每个神经元权重值数量
        layer.init_neurons(input, previous_neurons)
        self.layers.append(layer)
        # 下一层的权重值数量等于当前的神经元数量
        previous_neurons = input

        # hidden层初始化
        for i in range(len(hiddens)):  # 可能存在多层，这里只有一层
            layer = Layer()
            layer.init_neurons(hiddens[i], previous_neurons)
            self.layers.append(layer)
            previous_neurons = hiddens[i]

        # output层初始化，最后一层
        layer = Layer()
        layer.init_neurons(output, previous_neurons)
        self.layers.append(layer)

    # 获取权重值
    def get_weights(self):
        # network：神经网络结构，包含了层数，和各层的神经元数量，结果为[4, 16, 1]
        # weights：包含神经网络的所有权重值
        data = {'network': [], 'weights': []}
        for layer in self.layers:
            data['network'].append(len(layer.neurons))
            for neuron in layer.neurons:
                for weight in neuron.weights:
                    data['weights'].append(weight)
        return data

    def set_weights(self, data):
        previous_neurons = 0
        index_weights = 0
        self.layers = []  # 清空所有层，重新生成
        for i in data['network']:
            layer = Layer()
            layer.init_neurons(i, previous_neurons)
            for j in range(len(layer.neurons)):
                # 对该神经元的每一个权重值进行赋值
                # 第一层没有权重值，range值为0，循环直接跳过
                # ps：这里赋值的顺序和get_weights是对应的
                for k in range(len(layer.neurons[j].weights)):
                    layer.neurons[j].weights[k] = data['weights'][index_weights]
                    index_weights += 1
            previous_neurons = i  # 下一层权重值个数和当前神经元数量一样
            self.layers.append(layer)

    # 输入游戏环境中的一些条件(如敌机位置), 返回该神经网络的数学期望值
    def set_input_values(self, inputs):
        # 将这些条件赋值到输入层
        for i in range(len(inputs)):
            self.layers[0].neurons[i].value = inputs[i]
        # 将输入层作为上一层
        prev_layer = self.layers[0]
        for i in range(len(self.layers)):
            if i == 0:  # 第一层没有weights,直接跳过
                continue
            for j in range(len(self.layers[i].neurons)):  # 取出从隐藏层开始的每层的神经元个体
                sum = 0  # 该个体的数学期望值
                for k in range(len(prev_layer.neurons)):  # 取出上一层的神经元
                    # 上一层神经元的值和当前层的权重值乘机的累加和，为该个体的数学期望
                    sum += prev_layer.neurons[k].value * self.layers[i].neurons[j].weights[k]
                # 将该数学期望值映射到(0,1)范围内，赋值个当前神经元的值
                self.layers[i].neurons[j].value = sigmoid(sum)
            # 将本层作为下次循环的上一层，即下次输出层的一个神经元由隐藏层的16个神经元期望值*权重值的数学期望
            prev_layer = self.layers[i]
        # 整个循环结束后，输出层的数学期望就得到了
        # 将输出层的数学期望结果打包作为返回值
        out = []
        last_layer = self.layers[-1]
        # 因为输出层只有一个神经元，所以结果列表中只有一个数据
        for i in range(len(last_layer.neurons)):
            out.append(last_layer.neurons[i].value)
        return out


# "基因组"
class Genome():
    def __init__(self, score, network_weights):
        # 分数用来决定个体的优劣
        self.score = score
        # 个体基因包含该个体的所有神经元的权重值
        self.network_weights = network_weights


# 一代种群类
class Generation():
    def __init__(self):
        # 每一代包含很多个体基因
        self.genomes = []

    # 插入一个个体基因
    def insert_genome(self, genome):
        i = 0  # 第一次genomes列表长度为0，插入第一个
        # 遍历所有基因个体，将分数高的插入在前面，即在列表前面的基因是最优秀的
        for i in range(len(self.genomes)):
            if genome.score > self.genomes[i].score:
                break
        self.genomes.insert(i, genome)

    # 使用2个个体基因繁殖出下一代，这里繁殖一个(计划生育)
    def breed(self, genome1, genome2):
        # 繁殖的方式主要就是基因交叉+突变
        data = copy.deepcopy(genome1)  # 复制基因组
        for i in range(len(genome2.network_weights['weights'])):
            # 0.5概率交叉基因
            if random.random() <= 0.5:
                data.network_weights['weights'][i] = genome2.network_weights['weights'][i]

        for i in range(len(data.network_weights['weights'])):
            # 根据指定概率突变
            if random.random() <= mutation_rate:
                data.network_weights['weights'][i] += random_weight()
        return data

    # 生成下一代种群基因群体的权重值
    def create_next_generation_weights(self):
        nexts = []
        # 选取上一代中的精英，比率为elitism，因为在插入时已经做好了排序，这里前面几个就是分数最高的
        for i in range(round(elitism * population)):
            if len(nexts) < population:  # 下一代的数量不能超过种群数量
                # 下一代主要存储的是每个个体的权重值列表
                nexts.append(self.genomes[i].network_weights)

        # 创建一批随机行为的个体进入下一代
        for i in range(round(random_behaviour * population)):
            # 随便复制一个基因的权重值列表，比重新创建方便
            n = copy.deepcopy(self.genomes[0].network_weights)
            # 对权重值列表的所有值全部重新随机赋值
            for k in range(len(n['weights'])):
                n['weights'][k] = random_weight()
            if len(nexts) < population:
                nexts.append(n)

        # 剩余的进行交叉和突变
        while True:
            father = random.randint(0, len(self.genomes) - 1)
            mother = random.randint(0, len(self.genomes) - 1)
            if father == mother:
                continue
            child = self.breed(self.genomes[father], self.genomes[mother])
            nexts.append(child.network_weights)
            if len(nexts) == population:
                return nexts
        return nexts


# 种群世代的管理类
class GenerationManager(object):
    def __init__(self):
        self.generations = []  # 世代列表

    # 创建第一代种群的权重值
    def first_generation_weights(self):
        ret = []
        for i in range(population):
            nn = NeuroNetwork()
            # 初始化神经网络，包含了各层神经元和神经元权重值
            nn.init_neuro_network(network[0], network[1], network[2])
            # 提取权重值列表放入结果列表
            ret.append(nn.get_weights())
        # 生成了一代就需要在时代列表中增加一代
        self.generations.append(Generation())
        return ret

    # 根据上一代种群创建下一代种群的权重值
    def next_generation_weights(self):
        # 当前世代列表不能为空
        if len(self.generations) == 0:
            return False
        # 使用最后一代生成下一代权重值
        ret = self.generations[-1].create_next_generation_weights()
        # 生成了一代就需要在时代列表中增加一代
        self.generations.append(Generation())
        return ret

    # 为最新的一代添加新的基因（包含分数和权重值）
    def add_genome(self, genome):
        # 当前世代列表不能为空
        if len(self.generations) == 0:
            return False
        # 使用最后一代插入一个基因
        self.generations[-1].insert_genome(genome)


# 封装上述代码，对外部提供接口
class AI(object):
    def __init__(self):
        self.gen_manager = GenerationManager()

    # 创建下一代种群的神经网络
    def next_gen_NeuroNetwork(self):
        networks = []
        # 世代列表中没有数据，代表创建第一代种群
        if len(self.gen_manager.generations) == 0:
            networks = self.gen_manager.first_generation_weights()
        else:  # 否则利用上一代繁衍下一代种群
            networks = self.gen_manager.next_generation_weights()

        next_NeuroNetwork = []  # 下一代种群的神经网络列表
        for i in range(len(networks)):
            n = NeuroNetwork()
            # 这里只设置权重，value需要输入层的值进行计算，详见set_input_values
            n.set_weights(networks[i])
            next_NeuroNetwork.append(n)
        # 保存历史数量必须大于0，至少要保存一代繁衍下一代
        if historic > 0:
            # 当前种群世代列表超过保存值
            if len(self.gen_manager.generations) > historic:
                # 除去historic数量的记录，之前的全部删除，避免占用太多内存
                del self.gen_manager.generations[0:len(self.gen_manager.generations) - historic]

        return next_NeuroNetwork

    # 采集分数，增加新的基因
    def network_score(self, score, neuro_network):
        self.gen_manager.add_genome(Genome(score, neuro_network.get_weights()))
