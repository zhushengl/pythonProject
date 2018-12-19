import pygame
import random
import pygame.locals as locals
import simple.neuro_evolution as neuro_evolution

pygame.init()
FPS = 50
SCREEN_SIZE = (300, 400)
PIPE_GAP_SIZE = 100  # 管道上下之间的间隙
surface = pygame.display.set_mode(SCREEN_SIZE)


class Bird(object):
    def __init__(self,neuronetwork):
        self.img = pygame.image.load("./redbird-midflap.png")
        self.width = self.img.get_width()
        self.height = self.img.get_height()
        self.x = 100
        self.y = SCREEN_SIZE[1] / 2 - self.height / 2
        self.speed = 0
        self.neuronetwork = neuronetwork

    def update(self):
        self.y -= self.speed
        if self.speed > -7:
            self.speed -= 1
        surface.blit(self.img, (self.x, self.y))

    def fly(self):
        self.speed = 7

    def is_dead(self, pipes):
        if self.y < 0 or self.y > SCREEN_SIZE[1] - self.height:
            return True
        for pipe in pipes:
            # print((self.x, self.y), (pipe.x, pipe.upper_y, pipe.lower_y))
            if self.x + self.width > pipe.x \
                    and self.x < pipe.x + pipe.width:
                if self.y < pipe.upper_y \
                        or self.y + self.height > pipe.lower_y:
                    return True
        return False

    def get_input_values(self, pipe):
        inputs = [0.0, 0.0]
        # inputs[0] = self.y # 小鸟的y坐标
        if len(pipe):
            inputs[0] = (pipe[0].upper_y+pipe[0].lower_y)/2-self.y
            # inputs[2] = pipe[0].lower_y-5
            inputs[1] = self.x-pipe[0].x  # 小鸟到管道的距离
        ret = self.neuronetwork.feed_value(inputs)
        return ret


class Pipe(object):
    IMAGES = (
        pygame.transform.rotate(
            pygame.image.load("./pipe-green.png").convert_alpha(), 180),
        pygame.image.load("./pipe-green.png").convert_alpha(),
    )

    def __init__(self):
        self.width = Pipe.IMAGES[0].get_width()
        self.height = Pipe.IMAGES[0].get_height()
        self.upper_y = random.randint(30, (SCREEN_SIZE[1] - PIPE_GAP_SIZE - 30))
        self.lower_y = self.upper_y + PIPE_GAP_SIZE
        self.x = SCREEN_SIZE[0]

    def update(self):
        self.x -= 4
        surface.blit(Pipe.IMAGES[0], (self.x, self.upper_y - self.height))
        surface.blit(Pipe.IMAGES[1], (self.x, self.lower_y))

    def need_remove(self):
        if self.x < -self.width:
            return True
        return False


class Game(object):
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.ai = neuro_evolution.AI()
        self.gen = 0

    def start(self):
        self.birds = []
        self.pipes = []
        self.score = 0

        self.pipes.append(Pipe())
        self.gen_neuro_network_list = self.ai.next_generation_network_list()
        for i in range(len(self.gen_neuro_network_list)):
            self.birds.append(Bird(self.gen_neuro_network_list[i]))
        self.gen += 1
        self.alives = len(self.gen_neuro_network_list)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == locals.QUIT:
                    exit()
                # if event.type == locals.KEYDOWN and event.key == locals.K_SPACE:
                #     self.bird.fly()
            self.score += 1
            surface.fill((155, 155, 155))
            self.update_pipe()
            self.update_bird()

            print("alive:{}, generation:{}, score:{}".format(self.alives, self.gen, self.score), end='\n')
            pygame.display.update()
            self.clock.tick(FPS)

    def update_pipe(self):
        for pipe in self.pipes:
            if pipe.need_remove():
                self.pipes.remove(pipe)
            else:
                pipe.update()
        if self.score % 100 == 0:
            self.pipes.append(Pipe())

    def update_bird(self):
        for bird in self.birds:
            if not bird.is_dead(self.pipes):  # 如果小鸟存活
                # 输入 输入层数据,从神经网络获得 ret
                ret = bird.get_input_values(self.pipes)
                # 结果在0~1之间，以0.5作为正负分界，根据结果确定样本的移动方向
                if ret[0] > 0.5:  # 期望结果偏向正数，向上移动
                    bird.fly()
                bird.update()  # 更新birds进行移动和绘制
            else:
                self.birds.remove(bird)
                self.alives -= 1  # 记录当前存活样本数量，用于打印输出
                # 根据当前样本的基因和分数，将个体数据记录下来
                self.ai.gather_score(bird.neuronetwork, self.score)

        if len(self.birds) == 0:  # 样本全部死亡后，生成下一代重新开始
            self.start()


if __name__ == '__main__':
    game = Game()
    game.start()
    game.run()
