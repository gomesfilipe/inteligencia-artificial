import pygame
import os
import random
from sys import exit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"
RENDER_GAME = False
# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
if RENDER_GAME:
    SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))

class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
        pygame.draw.rect(SCREEN, self.color,
                         (self.dino_rect.x, self.dino_rect.y, self.dino_rect.width, self.dino_rect.height), 2)


    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)

class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))

class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)

class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345

class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325

class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1

class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight, nextObType, distanceGround):
        pass

    def updateState(self, state):
        pass

class DecisionTreeKeyClassifier(KeyClassifier):
    def __init__(self, state):
        self.alpha = state[0]
        self.beta = state[1]

    def __isGroundBird(self, obType, obHeight):
        if isinstance(obType, Bird) and obHeight < 40:
            return True
        return False
    
    def __shouldUp(self, obType, obHeight):
        if isinstance(obType, SmallCactus) or isinstance(obType, LargeCactus) or self.__isGroundBird(obType, obHeight):
            return True
        return False

    def keySelector(self, distance, obHeight, speed, obType, nextObDistance, nextObHeight,nextObType, distanceGround):
        limDistUp = speed * self.alpha # Distância para pular é diretamente proporcional a velocidade do jogo
        limDistDown = speed * self.beta # Distância para abaixar durante o pulo é diretamente proporcional a velocidade do jogo

        if distance <= limDistUp and distance >= limDistDown: # Perto de obstáculo
            if self.__shouldUp(obType, obHeight):
                return "K_UP"
            else:
                return "K_DOWN"
        else:
            return "K_DOWN"
            
    def updateState(self, state):
        self.state = state

def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"

def playGame(solutions):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True

    clock = pygame.time.Clock()
    cloud = Cloud()
    font = pygame.font.Font('freesansbold.ttf', 20)

    players = []
    players_classifier = []
    solution_fitness = []
    died = []

    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0

    obstacles = []
    death_count = 0
    spawn_dist = 0

    for solution in solutions:
        players.append(Dinosaur())
        players_classifier.append(DecisionTreeKeyClassifier(solution))
        solution_fitness.append(0)
        died.append(False)

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        if RENDER_GAME:
            text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)


    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed


    def statistics():
        text_1 = font.render(f'Dinosaurs Alive:  {str(died.count(False))}', True, (0, 0, 0))
        text_3 = font.render(f'Game Speed:  {str(game_speed)}', True, (0, 0, 0))

        SCREEN.blit(text_1, (50, 450))
        SCREEN.blit(text_3, (50, 480))

    while run and (False in died):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        if RENDER_GAME:
            SCREEN.fill((255, 255, 255))

        for i,player in enumerate(players):
            if not died[i]:
                distance = 1500
                nextObDistance = 2000
                obHeight = 0
                nextObHeight = 0
                obType = 2
                nextObType = 2
                if len(obstacles) != 0:
                    xy = obstacles[0].getXY()
                    distance = xy[0]
                    obHeight = obstacles[0].getHeight()
                    obType = obstacles[0]

                if len(obstacles) == 2:
                    nextxy = obstacles[1].getXY()
                    nextObDistance = nextxy[0]
                    nextObHeight = obstacles[1].getHeight()
                    nextObType = obstacles[1]

                xDino, yDino = players[i].getXY()
                groundDistance = 355 - yDino
                userInput = players_classifier[i].keySelector(distance, obHeight, game_speed, obType, nextObDistance, nextObHeight, nextObType, groundDistance)

                player.update(userInput)

                if RENDER_GAME:
                    player.draw(SCREEN)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        for obstacle in list(obstacles):
            obstacle.update()
            if RENDER_GAME:
                obstacle.draw(SCREEN)


        if RENDER_GAME:
            background()
            statistics()
            cloud.draw(SCREEN)

        cloud.update()

        score()

        if RENDER_GAME:
            clock.tick(60)
            pygame.display.update()

        for obstacle in obstacles:
            for i, player in enumerate(players):
                if player.dino_rect.colliderect(obstacle.rect) and died[i] == False:
                    solution_fitness[i] = points
                    died[i] = True

    return solution_fitness

class Particle:
    w = 0.08
    c1 = 0.2
    c2 = 0.05

    def __init__(self, position):
        self.__position = position
        self.__velocity = len(position) * [0]
        self.__bestLocal = self.__position
        self.__bestLocalValue = 0
    
    def updateVelocity(self, bestGlobal):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)

        t1 = self.__multParticle(self.__velocity, Particle.w)
        t2 = self.__multParticle(self.__subParticles(self.__bestLocal, self.__position), Particle.c1 * r1)
        t3 = self.__multParticle(self.__subParticles(bestGlobal.getPosition(), self.__position), Particle.c2 * r2)

        self.__velocity = self.__sum3Particles(t1, t2, t3)
        self.__velocity = [random.uniform(0, 10) if x > 50 else x for x in self.__velocity]
    
    def updatePosition(self):
        self.__position = self.__sum2Particles(self.__position, self.__velocity)
        self.__position = [random.uniform(0, 30) if x > 50 else x for x in self.__position]

    def updateBestLocal(self, particle, value):
        if value > self.__bestLocalValue:
            self.__bestLocalValue = value
            self.__bestLocal = Particle(particle.getPosition()).getPosition()
    
    def __str__(self):
        return f'Position: {[float("%.3f" % x) for x in self.__position]} // Velocity: {[float("%.3f" % x) for x in self.__velocity]}\n'
    
    def getPosition(self):
        return self.__position

    def getBestLocal(self):
        return self.__bestLocal
    
    def getBestLocalValue(self):
        return self.__bestLocalValue
    
    def __sum2Particles(self, a, b):
        return [a[i] + b[i] for i in range(len(a))]

    def __sum3Particles(self, a, b, c):
        return [a[i] + b[i] + c[i] for i in range(len(a))]

    def __subParticles(self, a, b):
        return [abs(a[i] - b[i]) for i in range(len(a))]

    def __multParticle(self, a, constant):
        return [constant * i for i in a]

class Swarm:
    def __init__(self, populationSize):
        # self.__population = [Particle([random.uniform(22, 24), random.uniform(4, 8)]) for x in range(populationSize)]
        self.__population = [Particle([random.uniform(0, 1), random.uniform(0, 1)]) for x in range(populationSize)]
        self.__bestGlobal = self.__population[0]
        self.__bestGlobalValue = 0

    def updateBestGlobal(self, particle, value):
        if value > self.__bestGlobalValue:
            self.__bestGlobalValue = value
            self.__bestGlobal = Particle(particle.getPosition())
            print(f'New best state! State: {self.__bestGlobal.getPosition()} // Points: {self.__bestGlobalValue}')

    def updateSwarm(self, particles, values):
        for i in range(len(self.__population)):
            self.__population[i].updateBestLocal(particles[i], values[i])
            self.updateBestGlobal(particles[i], values[i])
            self.__population[i].updateVelocity(self.__bestGlobal)
            self.__population[i].updatePosition()
    
    def getPopulation(self):
        return self.__population
    
    def getPopulationToList(self):
        return [particle.getPosition() for particle in self.__population]

    def getBestGlobal(self):
        return self.__bestGlobal

    def getBestGlobalValue(self):
        return self.__bestGlobalValue

    def __str__(self):
        output = ''
        for particle in self.__population:
            output += particle.__str__()
        return output

class PSO:
    def __init__(self, iterations, populationSize, file = None):
        self.__swarm = Swarm(populationSize)
        self.__iterations = iterations
        self.__file = file

    def execute(self):
        for i in range(self.__iterations):
            if i % 100 == 0:
                print(f'Population {i}')
            
            values = manyPlaysResultsTrain(5, self.__swarm.getPopulationToList())
            
            if self.__file != None:
                self.__file.write(f'{i + 1},{np.mean(values)},{max(values)}\n')

            population = self.__swarm.getPopulation()
            self.__swarm.updateSwarm(population, values)
        
        return self.__swarm.getBestGlobal().getPosition(), self.__swarm.getBestGlobalValue()

def manyPlaysResultsTrain(rounds,solutions):
    results = []

    for round in range(rounds):
        results += [playGame(solutions)]

    npResults = np.asarray(results)

    mean_results = np.mean(npResults,axis = 0) - np.std(npResults,axis=0) # axis 0 calcula media da coluna
    return mean_results

def manyPlaysResultsTest(rounds,best_solution):
    results = []
    for round in range(rounds):
        results += [playGame([best_solution])[0]]

    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())

def runsDataFrame(myResult, profResult):
  my = ['Student'] + myResult + [np.mean(myResult), np.std(myResult)]
  prof = ['Teacher'] + profResult + [np.mean(profResult), np.std(profResult)]

  columns = ['Person'] + [f'Run {i + 1}' for i in range(len(myResult))] + ['mean_runs', 'std_runs']

  return pd.DataFrame(data = [my, prof], columns = columns).transpose()

def boxplotDinos(myResult, profResult):
  sns.boxplot(data = [myResult, profResult]).set(xlabel='Aluno / Professor', ylabel='Pontuação')
  plt.savefig("boxplot.svg")
  plt.show()

def main():
    global aiPlayer
    
    # # Execução do PSO.
    # iterations = 10000
    # populationSize = 50

    # file = open('data.csv', 'w')

    # pso = PSO(iterations, populationSize, file)
    # bestState, bestValue = pso.execute()

    # file.close()

    # Melhor solução encontrada pelo PSO.
    bestState = [23.89132954117281, 6.298481701945017]

    aiPlayer = DecisionTreeKeyClassifier(bestState)
    myResult, value = manyPlaysResultsTest(30, bestState)
    
    profResult = [
        1214.0, 759.5, 1164.25, 977.25, 1201.0, 930.0, 1427.75, 799.5, 1006.25, 783.5,
        728.5, 419.25, 1389.5, 730.0, 1306.25, 675.5, 1359.5, 1000.25, 1284.5, 1350.0,
        751.0, 1418.75, 1276.5, 1645.75, 860.0, 745.5, 1426.25, 783.5, 1149.75, 1482.25
    ]

    # Informações das corridas do meu dino e o baseline.
    df = runsDataFrame(myResult, profResult)
    print('----------------------')
    print(df)
    print('----------------------\n')

    # Testes de hipótese
    s, p = ttest_rel(myResult, profResult)
    print(f'Teste T Pareado: {p}')

    s, p = wilcoxon(myResult, profResult)
    print(f'Teste de Wilcoxon: {p}')

    # Boxplot da distribuição das pontuações das 30 corridas.
    # Gera um arquivo SVG com a imagem do boxplot e também a mostra no terminal.
    boxplotDinos(myResult, profResult)

main()
