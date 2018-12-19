import pygame
import pygame.locals as locals
import random
import neuro_evolution as neuro_evolution


class Background(object):
    def __init__(self):
        self.y1 = 0
        self.y2 = -SCREEN_SIZE[1]
        self.img = pygame.image.load("./res/background.jpg")

    def update(self, surface):
        if self.y1 == SCREEN_SIZE[1]:
            self.y1 = -SCREEN_SIZE[1]
        if self.y2 == SCREEN_SIZE[1]:
            self.y2 = -SCREEN_SIZE[1]
        self.y1 += 4
        self.y2 += 4

        surface.blit(self.img, (0, self.y1))
        surface.blit(self.img, (0, self.y2))


class GameObject(object):
    def __init__(self, img_file):
        self.x = 0
        self.y = 0
        self.img = pygame.image.load(img_file).convert_alpha()
        self.width = self.img.get_width()
        self.height = self.img.get_height()
        self.is_alive = True

    def draw(self, surface):
        surface.blit(self.img, (self.x - self.width / 2, self.y - self.height / 2))


class Enemy(GameObject):
    def __init__(self):
        super().__init__("./res/enemy.png")
        self.speed = 12
        self.y = -self.height / 2
        min_x = self.width // 2
        max_x = SCREEN_SIZE[0] - self.width // 2
        self.x = random.randint(min_x, max_x)

    def update(self, surface):
        self.y += self.speed
        self.draw(surface)
        if self.y > SCREEN_SIZE[1] + self.height / 2:
            self.is_alive = False


class Plane(GameObject):
    def __init__(self):
        super().__init__("./res/plane.png")
        self.direction = 0  # 0,-1,1
        self.speed = 4
        self.x = SCREEN_SIZE[0] / 2
        self.y = SCREEN_SIZE[1] - self.height / 2
        self.min_x = 0
        self.max_x = SCREEN_SIZE[0]

    def update(self, surface):
        self.x += self.direction * self.speed
        super().draw(surface)

    def collision(self, enemy):
        if self.x - self.width / 2 < enemy.x + enemy.width / 2 and \
                self.x + self.width / 2 > enemy.x - enemy.width / 2 and \
                self.y - self.height / 2 < enemy.y + enemy.height / 2 and \
                self.y + self.height / 2 > enemy.y - enemy.height / 2:
            return True
        return False

    def is_dead(self, enemies):
        if self.x < self.min_x or self.x > self.max_x:
            return True
        for enemy in enemies:
            if self.collision(enemy):
                return True
        return False

    def get_input_values(self, enemes, size=4):
        # 先全部初始化为0
        inputs = []
        for i in range(size):
            inputs.append(0.0)
        # 第1数据为plane的x坐标，映射在（0，1上）
        inputs[0] = self.x * 1.0 / SCREEN_SIZE[0]
        # 第2，3数据为enemy的x，y坐标，如果当前不存在enemy，即为默认值0
        if len(enemes) > 0:
            inputs[1] = enemes[0].x * 1.0 / SCREEN_SIZE[0]
            inputs[2] = enemes[0].y * 1.0 / SCREEN_SIZE[1]
        # 第4数据，为plane相对于enemy的左右位置，取-1.0和1.0，2个数据
        if len(enemes) > 0 and self.x < enemes[0].x:
            inputs[3] = -1.0
        else:
            inputs[3] = 1.0
        return inputs


class Game(object):
    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        # self.bg = Background()
        pygame.display.set_caption('我会进化')
        self.ai = neuro_evolution.AI()
        self.generation = 0
        self.max_enemies = 1

    def start(self):
        self.score = 0
        self.enemies = []
        self.planes = []
        # 使用进化算法生成下一代神经网络
        self.gen_neuro_network_list = self.ai.next_gen_NeuroNetwork()
        for i in range(len(self.gen_neuro_network_list)):
            self.planes.append(Plane())
        self.generation += 1
        self.alives = len(self.gen_neuro_network_list)

    def run(self, FPS=200):
        while True:
            for event in pygame.event.get():
                if event.type == locals.QUIT:
                    exit()
            self.update()
            pygame.display.update()
            self.clock.tick(FPS)

    def update(self):
        # self.bg.update(self.surface)
        self.surface.fill((155,155,155))
        self.updateEnemies()
        self.updatePlanes()
        self.score += 1
        print("alive:{}, generation:{}, score:{}".format(self.alives, self.generation, self.score), end='\n')

    def updateEnemies(self):
        if len(self.enemies) < self.max_enemies:
            self.enemies.append(Enemy())
        for enemy in self.enemies:
            if enemy.is_alive:
                enemy.update(self.surface)
            else:
                self.enemies.remove(enemy)
                break

    def updatePlanes(self):
        for i in range(len(self.planes)):
            if self.planes[i].is_alive:
                # 获取输入层节点的数据
                input_values = self.planes[i].get_input_values(self.enemies)
                # 将数据传给个体样本的神经元输入层，并得到个体样本的预期结果
                ret = self.gen_neuro_network_list[i].set_input_values(input_values)
                # 结果在0~1之间，以0.5作为正负分界，根据结果确定样本的移动方向
                if ret[0] < 0.49:  # 期望结果偏向负数，向左移动
                    self.planes[i].direction = -1.0
                elif ret[0] > 0.51:  # 期望结果偏向正数，向右移动
                    self.planes[i].direction = 1.0
                else:  # 期望结果接近中心值，不移动
                    self.planes[i].direction = 0
                self.planes[i].update(self.surface)  # 更新plane进行移动和绘制
                if self.planes[i].is_dead(self.enemies):
                    self.planes[i].is_alive = False
                    self.alives -= 1  # 记录当前存活样本数量，用于打印输出
                    # 根据当前样本的分数和基因，将个体数据记录下来
                    self.ai.network_score(self.score, self.gen_neuro_network_list[i])
                    if self.is_all_dead():  # 样本全部死亡后，生成下一代重新开始
                        self.start()

    def is_all_dead(self):
        for plane in self.planes:
            if plane.is_alive:
                return False
        return True


if __name__ == '__main__':
    SCREEN_SIZE = (512, 700)
    game = Game()
    game.start()
    game.run()
