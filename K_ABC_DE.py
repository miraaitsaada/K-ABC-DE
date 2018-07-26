import numpy as np
import random
import k_means
import time

temps_execution=0

MAX_iter = 50
max_sans_amelioration = 25

limit = 50
SN = 100

def SSE(data, k, solution):
    somme_sse = 0
    for donnee in data:
        distance_minimale=np.linalg.norm(donnee-solution[0])
        for indice_centroid in range(1,k):
            distance = np.linalg.norm(donnee-solution[indice_centroid])
            if (distance < distance_minimale):
                distance_minimale = distance
        somme_sse += distance_minimale
    return somme_sse

def SSE1(data, k, solution):
    somme_sse = 0
    for donnee in data:
        distance_minimale = min([np.linalg.norm(donnee-i) \
                               for i in solution])
        somme_sse+=distance_minimale
    return somme_sse



def fit(data, k, solution):
    return 1/(1+SSE(data, k, solution))


def generer_solution(k, d, Xmin, Xmax):
    solution=np.empty((k,d))
    for i in range(k):
        for j in range(d):
            solution[i,j] = Xmin[j] + (random.uniform(0,1) * (Xmax[j]-Xmin[j]))
    return solution

def calcul_proba(fitness):
    proba = np.empty(SN)
    somme_fit = 0
    for fitness_empl in fitness:
        somme_fit += fitness_empl
    somme_proba = 0
    
    for i in range(SN):
        somme_proba += (fitness[i] / somme_fit)
        proba[i] = somme_proba
             
    return proba

def choix_onlooker1(population, fitness,proba):
    alea = random.uniform(0,1)
    j = 0
    for j in range(SN): #parcourir le tableau des probabilies
        if (alea < proba[j]):
            return j


def voisinage_employed(k , d , solution, indice_solution , population, Gbest, F1):
    V = np.empty((k,d))
    i3 = random.randint(0,SN-1)
    while i3 == indice_solution :
        i3 = random.randint(0,SN-1)
    i1 = random.randint(0,SN-1)
    while i1 == indice_solution :
        i1 = random.randint(0,SN-1)
    i2 = random.randint(0,SN-1)
    while i2 == indice_solution :
        i2 = random.randint(0,SN-1)
    
    for i in range(k):
        kij = random.uniform(0,1)
        for j in range(d):
            V[i,j]=solution[i,j]+ (kij * (population[i1,i,j] - solution[i,j])) + (F1*(population[i2,i,j] - population[i3,i,j]))
    return V

    
def voisinage_onlooker(k, d, solution, indice_solution, population, Gbest):
    #voisinage Global Best
    V=np.empty((k,d))
    g = random.randint(0,SN-1)
    while g == indice_solution :
        g = random.randint(0,SN-1)
    for i in range(k):
        Fi = random.uniform(-1 , 1)
        Psy = random.uniform(0 , 1.5)
        for j in range(d):
            V[i,j]=solution[i,j] + (Fi*(solution[i,j]-population[g,i,j])) + (Psy * (Gbest[i,j] - solution[i,j]))
        
    return V


def mutation(k, d, solution, indice_solution, population, Gbest):
    V = np.empty((k,d))
    
    k1 = random.randint(0,SN-1)
    while(k1 == indice_solution):
        k1 = random.randint(0,SN-1)
    k2 = random.randint(0,SN-1)
    while(k2 == indice_solution):
        k2 = random.randint(0,SN-1)
    
    for i in range(k):
        for j in range(d):
            V[i,j]=(random.uniform(0,1)*(solution[i,j]-population[k1,i,j])) + (random.uniform(0,1)*(Gbest[i,j]-population[k2,i,j]))
    return V


def K_ABC_DE(data, k):
        
        global temps_execution
        
        essais = np.zeros(SN, dtype=int)
        
        d=np.size(data[0])
        
        Xmin = np.empty(d)
        Xmax = np.empty(d)

        for i in range(d):
            Xmax[i] = np.amax(data.T[i])
            Xmin[i] = np.amin(data.T[i])
            
        population=np.empty((SN,k,d));
        fitness=np.empty((SN))

        Gbest = np.array(k_means.k_means(data.tolist(), k)) # generer_solution(k, d, Xmin, Xmax)

        Fbest = fit(data, k, Gbest)
        
        
        for i in range(SN):
            solution = generer_solution(k, d, Xmin, Xmax)
            fitness_solution = fit(data, k, solution)
            population[i] = solution
            fitness[i] = fitness_solution
        
            if (fitness_solution > Fbest):
                Gbest = solution
                Fbest = fitness_solution
            
        
        cycle = 0
        sans_amelioration = 0
        
        start_time = time.time()
        
        while (cycle < MAX_iter and sans_amelioration < 200): 
            
            F1 = random.uniform(0,1)
            
            #phase des employes :
            for i, solution_courante in enumerate(population):
                    fitness_solution_courante = fitness[i]
                    
                    voisine = voisinage_employed(k , d , solution_courante , i, population, Gbest,F1)
                    fitness_voisine = fit(data, k, voisine)
                    if (fitness_voisine > fitness_solution_courante):
                        population[i] = voisine
                        fitness[i] = fitness_voisine
                        essais[i] = 0
                        sans_amelioration = 0
                        if (fitness_voisine > Fbest):
                            Gbest = voisine
                            Fbest = fitness_voisine
                        else:
                            essais[i] += 1
                            sans_amelioration +=1
            
                    
            #calcul des probabilies
            proba = np.empty(SN)
            proba = calcul_proba(fitness)
        
            
            #phase des onlookers
            for i in range(SN):
                #sÄĹ lÄĹ ection d'une solution
                indice_choisi = choix_onlooker1(population,fitness,proba)
                onlooker_courant = population[indice_choisi]
                fitness_onlooker_courant = fitness[indice_choisi]
                
                #Exploitation de la solution
                voisine = voisinage_onlooker(k , d , onlooker_courant, indice_choisi , population, Gbest)
                fitness_voisine = fit(data, k, voisine)
                
                if (fitness_onlooker_courant < fitness_voisine):
                    population[indice_choisi] = voisine
                    fitness[indice_choisi] = fitness_voisine
                    essais[indice_choisi] = 0
                    sans_amelioration = 0
                    if (fitness_voisine > Fbest):
                        Fbest = fitness_voisine
                        Gbest = voisine
                else:
                    essais[indice_choisi] += 1
                    sans_amelioration += 1

            
            #Phase mutation
            for i in range(SN):
                S_new = mutation(k, d, solution, i, population, Gbest)
                F_new = fit(data, k, S_new)
                 
                if (fitness[i] < F_new):
                    population[i] = S_new
                    fitness[i] = F_new
                    essais[i] = 0
                    sans_amelioration = 0
                    if (F_new > Fbest):
                        Fbest = F_new
                        Gbest = S_new
                else:
                    essais[i] += 1
                    sans_amelioration += 1
                
            
            
            #Phase scout
            for i in range(SN):
                if (essais[i]>limit):
                    essais[i] = 0
                    nouvelle_solution = generer_solution(k, d, Xmin, Xmax)
                    population[i] = nouvelle_solution
                    nouvelle_fitness = fit(data, k, nouvelle_solution)
                    fitness[i] = nouvelle_fitness
                    if (nouvelle_fitness > Fbest):
                        Gbest = nouvelle_solution
                        Fbest = nouvelle_fitness
                    
            
            cycle += 1
            
        end_time = time.time()
        temps_execution = (end_time - start_time)
        
        return Gbest
