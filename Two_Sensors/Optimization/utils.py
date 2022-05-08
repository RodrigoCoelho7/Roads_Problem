import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import json
Binomial = tfp.distributions.Binomial

 
f = open(r'C:\Users\reasc\OneDrive - Universidade do Minho (1)\Mestrado\Segundo Semestre\CCF\Roads_Problem\Two_Sensors\Optimization\flow.json')
flow = json.load(f)

class CAR:
    def __init__(self,car):
        self.route = car['route']
        self.time = car['startTime'] 

class SEMAFORO:
    def __init__(self,T):
        self.time = T
        self.last_time = T
        self.evolution_time = 0
        self.state = False
        self.changing = False
        self.states = []
        self.evolution_times = []

    def evolve_step(self,semaforo_change):
        self.states.append([self.state,self.changing])#('Green' if self.state else 'Yellow' if self.changing else 'Red')
        self.evolution_times.append([self.time,self.evolution_time])
        if self.state:
            if self.evolution_time >= self.last_time:
                self.last_time = self.time
                self.state = False
                self.evolution_time = 1
                return True
            else:
                self.evolution_time += 1
                return False
        if semaforo_change or self.changing:
            self.changing = True
            if self.evolution_time == 6:
                self.state = True
                self.changing = False
                self.evolution_time = 1
            else:
                self.evolution_time += 1
        return False

    def change_time(self,t):
        self.last_time = self.time
        self.time = t

class ROAD:
    def __init__(self,Tlow,THigh,crossing_time):
        self.cars = []
        self.count_cars = 0
        self.semaforo = SEMAFORO(Tlow)
        self.time = 1
        self.crossing_time = crossing_time
        self.total_cars = []
        self.actual_cars = []
        self.Tlow = Tlow
        self.Thigh = THigh
    
    def append_car(self,car):
        self.cars.append(car)
    
    def generate_distribution(self):
        self.times = [car.time for car in self.cars]
        counts,bins=np.histogram(self.times,bins=int(3600/(60*3)))
        self.distribution = counts/(3*60)

    def generate_cars(self,time):
        i = time//(60*3)
        #print(time,i,self.distribution[i],len(self.distribution))
        self.semaforo.change_time(self.Tlow if self.distribution[i]*180 <= 50 else self.Thigh)
        cars = Binomial(1.,probs=self.distribution[i])
        car = cars.sample()
        self.count_cars += car
        self.total_cars.append(car)
        self.actual_cars.append(self.count_cars)

    def evolve_step(self,time,semaforo_change):
        self.generate_cars(time)
        semaforo_change = self.semaforo.evolve_step(semaforo_change)
        if self.semaforo.state:
            if self.time == self.crossing_time:
                self.count_cars -= 1 if self.count_cars > 0 else 0
                self.time = 1
            else:
                self.time += 1
        return semaforo_change
    
    def print_road_state(self):
        print(f'    COUNT_CARS: {self.count_cars}')
        semaforo = 'Green' if self.semaforo.state else 'Yellow' if self.semaforo.changing else 'Red'
        print(f'    SEMAFORO_STATE: {semaforo}')

class INTERSECTION:
    def __init__(self,Tlow,THigh):
        self.global_time = 0
        self.roads = [ROAD(Tlow,THigh,2),ROAD(Tlow,THigh,2)]
        for car in flow:
            c = CAR(car)
            if c.route[0] == 'road_0_1_0' or c.route[0] == 'road_2_1_2':
                self.roads[0].append_car(c)
            if c.route[0] == 'road_1_0_1' or c.route[0] == 'road_1_2_3':
                self.roads[1].append_car(c)
        self.semaforo_change = False
        [road.generate_distribution() for road in self.roads]
        self.roads[np.random.randint(0,2)].semaforo.state = True

    def step_system(self):
        for road in self.roads:
            self.semaforo_change = road.evolve_step(self.global_time,self.semaforo_change)
            #road.print_road_state()
        self.global_time += 1

    def time_evolution(self,Tmax):
        print('Start Evolution')
        for t in range(Tmax):
            print(f'TIME: {t+1}',end='\r')
            self.step_system()


if __name__ == '__main__':
    E = INTERSECTION(30,60)
    E.time_evolution(3600)
    bins = np.arange(0,3600+180,180)
    bincentres = [(bins[i]+bins[i+1])/2. for i in range(len(bins)-1)]
    fig,ax = plt.subplots()
    for i,road in enumerate(E.roads):
        total_cars = [np.sum(road.total_cars[j*180:(j+1)*180]) for j in range(20)]
        ax.step(bincentres,total_cars,label=f'Road{i+1}')  
    ax.set_title('Total Cars')
    ax.legend()

    fig2,ax2 = plt.subplots()
    for i,road in enumerate(E.roads):
        ax2.plot(road.actual_cars,label=f'Road{i+1}')  
    ax2.set_title('Actual Cars')
    ax2.legend()

    fig3,ax3 = plt.subplots()
    for i,road in enumerate(E.roads):
        semaforos = np.array(road.semaforo.states)*1#[0 if state=='Green' else 1 if state =='Yellow' else 2 for state in road.semaforo.states]
        ax3.plot(semaforos[:,0],label=f'Road{i+1}')  
        ax3.plot(semaforos[:,1],label=f'Road{i+1}')  
    ax3.set_title('Semaforo')
    ax3.legend()

    fig4,ax4 = plt.subplots()
    for i,road in enumerate(E.roads):
        semaforos = np.array(road.semaforo.evolution_times)
        ax4.plot(semaforos[:,0],label=f'Road{i+1}')  
        ax4.plot(semaforos[:,1],label=f'Road{i+1}')  
    ax4.set_title('Semaforo Times')
    ax4.legend()

    plt.show()