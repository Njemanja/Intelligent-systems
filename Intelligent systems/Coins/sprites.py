import math
import random
import sys
import pygame
import os
import numpy as np
import config
from itertools import permutations

class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, x, y, file_name, transparent_color=None, wid=config.SPRITE_SIZE, hei=config.SPRITE_SIZE):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (wid, hei))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)


class Surface(BaseSprite):
    def __init__(self):
        super(Surface, self).__init__(0, 0, 'terrain.png', None, config.WIDTH, config.HEIGHT)


class Coin(BaseSprite):
    def __init__(self, x, y, ident):
        self.ident = ident
        super(Coin, self).__init__(x, y, 'coin.png', config.DARK_GREEN)

    def get_ident(self):
        return self.ident

    def position(self):
        return self.rect.x, self.rect.y

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.BLACK)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class CollectedCoin(BaseSprite):
    def __init__(self, coin):
        self.ident = coin.ident
        super(CollectedCoin, self).__init__(coin.rect.x, coin.rect.y, 'collected_coin.png', config.DARK_GREEN)

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.RED)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class Agent(BaseSprite):
    def __init__(self, x, y, file_name):
        super(Agent, self).__init__(x, y, file_name, config.DARK_GREEN)
        self.x = self.rect.x
        self.y = self.rect.y
        self.step = None
        self.travelling = False
        self.destinationX = 0
        self.destinationY = 0

    def set_destination(self, x, y):
        self.destinationX = x
        self.destinationY = y
        self.step = [self.destinationX - self.x, self.destinationY - self.y]
        magnitude = math.sqrt(self.step[0] ** 2 + self.step[1] ** 2)
        self.step[0] /= magnitude
        self.step[1] /= magnitude
        self.step[0] *= config.TRAVEL_SPEED
        self.step[1] *= config.TRAVEL_SPEED
        self.travelling = True

    def move_one_step(self):
        if not self.travelling:
            return
        self.x += self.step[0]
        self.y += self.step[1]
        self.rect.x = self.x
        self.rect.y = self.y
        if abs(self.x - self.destinationX) < abs(self.step[0]) and abs(self.y - self.destinationY) < abs(self.step[1]):
            self.rect.x = self.destinationX
            self.rect.y = self.destinationY
            self.x = self.destinationX
            self.y = self.destinationY
            self.travelling = False

    def is_travelling(self):
        return self.travelling

    def place_to(self, position):
        self.x = self.destinationX = self.rect.x = position[0]
        self.y = self.destinationX = self.rect.y = position[1]

    # coin_distance - cost matrix
    # return value - list of coin identifiers (containing 0 as first and last element, as well)
    def get_agent_path(self, coin_distance):
        pass


class ExampleAgent(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        path = [i for i in range(1, len(coin_distance))]
        random.shuffle(path)
        return [0] + path + [0]

class Aki(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)
    def get_agent_path(self, coin_distance):
        path = []
        q=len(coin_distance)
        i=0
        while(q):
            min = 100000
            poz = 0
            for j in range(0, len(coin_distance)):
                if (coin_distance[i][j] < min and coin_distance[i][j] > 0 and j != 0):
                    if not j in path:
                        min = coin_distance[i][j]
                        poz = j

            path.append(poz)
            i=poz
            q=q-1
        """for i in range(0, len(coin_distance)):
            min = 100000
            poz = 0
            for j in range(0, len(coin_distance)):
                if (coin_distance[i][j] < min and coin_distance[i][j] > 0 and j != 0):
                    if not j in path:
                        min = coin_distance[i][j]
                        poz = j

            path.append(poz)
        print(coin_distance)
        print(path)"""
        return [0]+ path

class Jocke(Agent):

    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        lista=[]
        for i in range(1, len(coin_distance)):
            lista.append(i)

        perms= permutations(lista)
        perms=list(perms)
        l=[]
        for perm in perms:
            perm=list(perm)
            perm.append(0)
            perm.insert(0,0)
            l.append(perm)
        #print(l)
        first=True
        for perm in l:
            sum=0
            for i in range (0,len(perm)-1):
                sum=sum+coin_distance[perm[i]][perm[i+1]]
            #print(sum)
            if(first):
                min=sum
                first=False
            if(sum<min):
                min=sum
                p=perm
        #print(min)
        #print(p)
        return p


class Uki(Agent):

    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        lista = []
        for i in range(1, len(coin_distance)):
            lista.append(i)
        matrica=[]
        for l in lista:
            podatak=[]

            podatak.append(0)
            podatak.append(l)
            podatak.append(coin_distance[0][l])
            matrica.append(podatak)
        matrica=sorted(matrica, key=lambda x: x[-1])

        q=True
        while(q):
            najbolji=[]
            pocetak=True
            kraj=True
            #print(matrica)
            while(kraj):
                if(pocetak):
                    pocetak=False
                    najbolji2=matrica.pop(0)
                    najbolji.append(najbolji2)
                else:
                    najbolji2=matrica.pop(0)

                    if(najbolji[0][-1]==najbolji2[-1]):
                        najbolji.append(najbolji2)
                    else:
                        matrica.append(najbolji2)
                        kraj=False
            najbolji=sorted(najbolji, key=len, reverse=True)
            #print(najbolji)
            #print("\n")
            for i in range(1, len(najbolji)):
                matrica.append(najbolji[i])
            najbolji=najbolji[0]
            if(len(najbolji)==(len(lista)+3)):
                break
            tezina=najbolji.pop()
            for i in range(0,len(lista)):
                podatak=[]
                for j in range(0,len(najbolji)):
                    podatak.append(najbolji[j])
                if not (lista[i] in najbolji):
                    podatak.append(lista[i])
                    tezina1=tezina+coin_distance[najbolji[-1]][lista[i]]
                    podatak.append(tezina1)

                    if(len(podatak)==(len(lista)+2)):
                        t=podatak.pop()
                        t=t+coin_distance[podatak[-1]][0]
                        podatak.append(0)
                        podatak.append(t)

                    matrica.append(podatak)
                    matrica = sorted(matrica, key=lambda x: x[-1])

        #matrica = sorted(matrica, key=lambda x: x[-1])
        #print(najbolji)
        najbolji.pop()
        return najbolji


class Micko(Agent):
    def __init__(self, x, y, file_name):
        super().__init__(x, y, file_name)

    def get_agent_path(self, coin_distance):
        lista = []  # IZMENITI
        g = Graph(len(coin_distance))
        g.graph = coin_distance
        summ = g.primMST(lista)
        #print(summ)
        lista = []
        for i in range(1, len(coin_distance)):
            lista.append(i)
        matrica = []
        for l in lista:
            podatak = []
            podatak.append(0)
            podatak.append(l)
            podatak.append(summ)
            podatak.append(coin_distance[0][l])
            podatak.append(summ+coin_distance[0][l])
            matrica.append(podatak)
        matrica = sorted(matrica, key=lambda x: x[-1])

        #print(matrica)
        q = 20
        while (True):
            najbolji = []
            pocetak = True
            kraj = True
            # print(matrica)
            while (kraj):
                if (pocetak):
                    pocetak = False
                    najbolji2 = matrica.pop(0)
                    najbolji.append(najbolji2)
                else:
                    najbolji2 = matrica.pop(0)

                    if (najbolji[0][-1] == najbolji2[-1]):
                        najbolji.append(najbolji2)
                    else:
                        matrica.append(najbolji2)
                        kraj = False
            najbolji=sorted(najbolji)
            najbolji = sorted(najbolji, key=len, reverse=True)
            #print(najbolji)
            #print("\n")
            for i in range(1, len(najbolji)):
                matrica.append(najbolji[i])
            najbolji = najbolji[0]


            if (len(najbolji) == (len(lista) + 5)):
                break

            tezinaIheur=najbolji.pop()
            tezina = najbolji.pop()
            heur=najbolji.pop()
            #print(najbolji)
            for i in range(0, len(lista)):
                podatak = []
                listaObidjenih=[]
                for j in range(0, len(najbolji)):
                    podatak.append(najbolji[j])
                for j in range (1,len(podatak)):
                    listaObidjenih.append(najbolji[j])
                mat=coin_distance
                for j in range(0,len(listaObidjenih)):
                    #print(listaObidjenih[j])
                    mat=np.delete(mat,listaObidjenih[j],0)
                    mat=np.delete(mat, listaObidjenih[j] , 1)
                    for q in range (j+1,len(listaObidjenih)):
                        if(listaObidjenih[j]<listaObidjenih[q]):
                            listaObidjenih[q]=listaObidjenih[q]-1
                #print(mat)
                g = Graph(len(mat))
                g.graph = mat
                summ = g.primMST(lista)
                #print(summ)
                #print(lista)
                #print(i)
                #print(najbolji)
                if not (lista[i] in najbolji):
                    podatak.append(lista[i])
                    tezina1 = tezina + coin_distance[najbolji[-1]][lista[i]]
                    podatak.append((summ))
                    podatak.append(tezina1)
                    podatak.append(summ+tezina1)
                    #print(podatak)
                    if (len(podatak) == (len(lista) + 4)):
                        podatak.pop()
                        podatak.pop()
                        podatak.pop()
                        #t = t + coin_distance[podatak[-1]][0]
                        podatak.append(0)
                        return podatak
                        #podatak.append(t)
                    matrica.append(podatak)
                    matrica = sorted(matrica, key=lambda x: x[-1])
                    #print(matrica)

        return [0,1,2,3,4,0]





class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printMST(self, parent, lista):
        #print("Edge \tWeight")
        summ = 0
        l = [0,1, 2, 3, 4]  # IZMENITI
        for i in range(1, self.V):
            if (len(lista)==0):
                #print(l[parent[i]], "-", l[i], "\t", self.graph[i][parent[i]])
                summ = summ + self.graph[i][parent[i]]
            elif (i >= lista[0]):
                a = i + 1
                #print(l[parent[i]], "-", l[i], "\t", self.graph[i][parent[i]])
                summ = summ + self.graph[i][parent[i]]
            else:
                #print(l[parent[i]], "-", l[i], "\t", self.graph[i][parent[i]])
                summ = summ + self.graph[i][parent[i]]

        return summ

    def minKey(self, key, mstSet):
        min = sys.maxsize
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index

    def primMST(self, lista):
        key = [sys.maxsize] * self.V
        parent = [None] * self.V
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1

        for cout in range(self.V):
            u = self.minKey(key, mstSet)
            mstSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u

        return self.printMST(parent, lista)




