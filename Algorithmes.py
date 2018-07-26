import numpy as np
import random
import k_means


MAX_FE = 10000

limit = 50
SN = 100

pop_size = 50

F=0.6
CR=0.9

nb_appel_obj = 0


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
    global nb_appel_obj
    nb_appel_obj +=1
    return 1/(1+SSE(data, k, solution))


def generer_solution(k, d, Xmin, Xmax):
    solution=np.empty((k,d))
    for i in range(k):
        for j in range(d):
            solution[i,j] = Xmin[j] + (random.uniform(0,1) * (Xmax[j]-Xmin[j]))
    return solution

    
def voisinage(k, d, solution, indice_solution, population, Gbest, fitness_solution_courante, nb_essais):
    #voisinage d'origine (basique)
    V=np.empty((k,d))
    g = random.randint(0,SN-1)
    while g == indice_solution :
        g = random.randint(0,SN-1)
    for i in range(k):
        Fi = random.uniform(-1 , 1)
        for j in range(d):
            V[i,j]=solution[i,j] + (Fi*(solution[i,j]-population[g,i,j]))
    return V

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
    for j in range(SN): #parcourir le tableau des probabilites
        if (alea < proba[j]):
            return j
        
        
def voisinage_gabc(k, d, solution, indice_solution, population, Gbest, fitness_solution_courante, nb_essais):
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

def binomial(k, d, mutant, parent):
    cross = np.empty((k,d))
    j0 = random.randint(0,k-1)
    
    for j in range(k):
        if ((random.uniform(0,1) <= CR) or (j == j0)):
            cross[j] = mutant[j]
        else:
            cross[j] = parent[j]
    return cross


def voisinage_employed(k , d , solution, indice_solution , population, Gbest,fitness_onlooker_courant, essais, fitness, F1):
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



    
def voisinage_onlooker(k, d, solution, indice_solution, population, Gbest, fitness_solution_courante, nb_essais,fitness):
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


def current_to_rand1(k, d, indice_solution, population, Gbest, F1):
    V = np.empty((k,d))
    pop_size = population.shape[0]
    
    i = indice_solution
    
    i1 = random.randint(0,pop_size-1)
    while i1 == indice_solution :
        i1 = random.randint(0,pop_size-1)
    i2 = random.randint(0,pop_size-1)
    while i2 == indice_solution :
        i2 = random.randint(0,pop_size-1)
    i3 = random.randint(0,pop_size-1)
    while i3 == indice_solution :
        i3 = random.randint(0,pop_size-1)
    
    for g in range(k):
        kij = random.uniform(0,1)
        for j in range(d):
            V[g,j] = population[i,g,j]+(kij*(population[i3,g,j]-population[i,g,j]))+(F1*(population[i1,g,j]-population[i2,g,j]))
    return V
    
    


def rand1(k, d, indice_solution, population, Gbest, F):
    U=np.empty((k,d))
    pop_size = population.shape[0]
    
    i3 = random.randint(0,pop_size-1)
    while i3 == indice_solution :
        i3 = random.randint(0,pop_size-1)
    i1 = random.randint(0,pop_size-1)
    while i1 == indice_solution :
        i1 = random.randint(0,pop_size-1)
    i2 = random.randint(0,pop_size-1)
    while i2 == indice_solution :
        i2 = random.randint(0,pop_size-1)
    
    for g in range(k):
        for j in range(d):
            U[g,j] = population[i3,g,j] + (F*(population[i1,g,j]-population[i2,g,j]))
    return U


# ABC ----------------------------------------------------------------------


def ABC(data, k):
        
        global nb_appel_obj
        
        nb_appel_obj = 0
        essais = np.zeros(SN, dtype=int)
        
        d=np.size(data[0])
        
        Xmin = np.empty(d)
        Xmax = np.empty(d)

        for i in range(d):
            Xmax[i] = np.amax(data.T[i])
            Xmin[i] = np.amin(data.T[i])
            
        population=np.empty((SN,k,d))
        fitness=np.empty((SN))
        
        
        Gbest = generer_solution(k, d, Xmin, Xmax)
        Fbest = fit(data, k, Gbest)
        
        for i in range(SN):
            solution = generer_solution(k, d, Xmin, Xmax)
            fitness_solution = fit(data, k, solution)
            population[i] = solution
            fitness[i] = fitness_solution
        
            if (fitness_solution > Fbest):
                Gbest = solution
                Fbest = fitness_solution
            
            
        while (nb_appel_obj <= MAX_FE):
            
            #phase des employees :
            for i, solution_courante in enumerate(population):
                    fitness_solution_courante = fitness[i]
                    voisine = voisinage(k , d , solution_courante , i, population, Gbest, fitness_solution_courante, essais[i])
                    fitness_voisine = fit(data, k, voisine)
                    if (fitness_voisine > fitness_solution_courante):
                        population[i] = voisine
                        fitness[i] = fitness_voisine
                        essais[i] = 0
                        if (fitness_voisine > Fbest):
                            Gbest = voisine
                            Fbest = fitness_voisine
                        else:
                            essais[i] += 1
            
                    
            #calcul des probabilite
            proba = np.empty(SN)
            proba = calcul_proba(fitness)
        
            
            #phase des onlookers
            for i in range(SN):
                #selection d'une solution
                indice_choisi = choix_onlooker1(population,fitness,proba)
                onlooker_courant = population[indice_choisi]
                fitness_onlooker_courant = fitness[indice_choisi]
                
                #Exploitation de la solution
                voisine = voisinage(k , d , onlooker_courant, indice_choisi , population, Gbest,fitness_onlooker_courant, essais[indice_choisi])
                fitness_voisine = fit(data, k, voisine)
                
                if (fitness_onlooker_courant < fitness_voisine):
                    population[indice_choisi] = voisine
                    fitness[indice_choisi] = fitness_voisine
                    essais[indice_choisi] = 0
                    if (fitness_voisine > Fbest):
                        Fbest = fitness_voisine
                        Gbest = voisine
                else:
                    essais[indice_choisi] += 1
            
            
            #Phase scout
            for i in range(SN):
                if (essais[i]>limit):
                    essais[i]=0
                    nouvelle_solution = generer_solution(k, d, Xmin, Xmax)
                    population[i] = nouvelle_solution
                    nouvelle_fitness = fit(data, k, nouvelle_solution)
                    fitness[i] = nouvelle_fitness
                    if (nouvelle_fitness > Fbest):
                        Gbest = nouvelle_solution
                        Fbest = nouvelle_fitness
             
            
        return Gbest









# GABC---------------------------------------------------------------------------

def GABC(data, k):
        
        global nb_appel_obj
        
        nb_appel_obj = 0
        
        essais = np.zeros(SN, dtype=int)
        
        d=np.size(data[0])
        
        Xmin = np.empty(d)
        Xmax = np.empty(d)

        for i in range(d):
            Xmax[i] = np.amax(data.T[i])
            Xmin[i] = np.amin(data.T[i])
            
        population=np.empty((SN,k,d))
        fitness=np.empty((SN))
        
        Gbest = generer_solution(k, d, Xmin, Xmax)
        Fbest = fit(data, k, Gbest)
        
        for i in range(SN):
            solution = generer_solution(k, d, Xmin, Xmax)
            fitness_solution = fit(data, k, solution)
            population[i] = solution
            fitness[i] = fitness_solution
        
            if (fitness_solution > Fbest):
                Gbest = solution
                Fbest = fitness_solution
            
        
        while (nb_appel_obj <= MAX_FE):
            
            #phase des employees :
            for i, solution_courante in enumerate(population):
                    fitness_solution_courante = fitness[i]
                    voisine = voisinage_gabc(k , d , solution_courante , i, population, Gbest, fitness_solution_courante, essais[i])
                    fitness_voisine = fit(data, k, voisine)
                    if (fitness_voisine > fitness_solution_courante):
                        population[i] = voisine
                        fitness[i] = fitness_voisine
                        essais[i] = 0
                        if (fitness_voisine > Fbest):
                            Gbest = voisine
                            Fbest = fitness_voisine
                        else:
                            essais[i] += 1
            
                    
            #calcul des probabilites
            proba = np.empty(SN)
            proba = calcul_proba(fitness)
        
            
            #phase des onlookers
            for i in range(SN):
                #selection d'une solution
                indice_choisi = choix_onlooker1(population,fitness,proba)
                onlooker_courant = population[indice_choisi]
                fitness_onlooker_courant = fitness[indice_choisi]
                
                #Exploitation de la solution
                voisine = voisinage_gabc(k , d , onlooker_courant, indice_choisi , population, Gbest,fitness_onlooker_courant, essais[indice_choisi])
                fitness_voisine = fit(data, k, voisine)
                
                if (fitness_onlooker_courant < fitness_voisine):
                    population[indice_choisi] = voisine
                    fitness[indice_choisi] = fitness_voisine
                    essais[indice_choisi] = 0
                    if (fitness_voisine > Fbest):
                        Fbest = fitness_voisine
                        Gbest = voisine
                else:
                    essais[indice_choisi] += 1
            
            
            #Phase scout
            for i in range(SN):
                if (essais[i]>limit):
                    essais[i]=0
                    nouvelle_solution = generer_solution(k, d, Xmin, Xmax)
                    population[i] = nouvelle_solution
                    nouvelle_fitness = fit(data, k, nouvelle_solution)
                    fitness[i] = nouvelle_fitness
                    if (nouvelle_fitness > Fbest):
                        Gbest = nouvelle_solution
                        Fbest = nouvelle_fitness
             
    
        return Gbest







#ABC_DE -----------------------------------------------------------------

def ABC_DE(data, k):
        
        global nb_appel_obj
        
        nb_appel_obj = 0
        essais = np.zeros(SN, dtype=int)
        
        d=np.size(data[0])
        
        Xmin = np.empty(d)
        Xmax = np.empty(d)

        for i in range(d):
            Xmax[i] = np.amax(data.T[i])
            Xmin[i] = np.amin(data.T[i])
            
        population=np.empty((SN,k,d));
        fitness=np.empty((SN))
        
        
        Gbest = generer_solution(k, d, Xmin, Xmax)
        Fbest = fit(data, k, Gbest)
        
        for i in range(SN):
            solution = generer_solution(k, d, Xmin, Xmax)
            fitness_solution = fit(data, k, solution)
            population[i] = solution
            fitness[i] = fitness_solution
        
            if (fitness_solution > Fbest):
                Gbest = solution
                Fbest = fitness_solution
            
        
        while (nb_appel_obj <= MAX_FE):
            F1 = random.uniform(0,1)
            
            #phase des employes :
            for i, solution_courante in enumerate(population):
                    fitness_solution_courante = fitness[i]
                    
                    voisine = voisinage_employed(k , d , solution_courante , i, population, Gbest, fitness_solution_courante, essais[i],fitness,F1)
                    fitness_voisine = fit(data, k, voisine)
                    if (fitness_voisine > fitness_solution_courante):
                        population[i] = voisine
                        fitness[i] = fitness_voisine
                        essais[i] = 0
                        if (fitness_voisine > Fbest):
                            Gbest = voisine
                            Fbest = fitness_voisine
                        else:
                            essais[i] += 1
            
                    
            #calcul des probabilités
            proba = np.empty(SN)
            proba = calcul_proba(fitness)
        
            
            #phase des onlookers
            for i in range(SN):
                #selection d'une solution
                indice_choisi = choix_onlooker1(population,fitness,proba)
                onlooker_courant = population[indice_choisi]
                fitness_onlooker_courant = fitness[indice_choisi]
                
                #Exploitation de la solution
                voisine = voisinage_onlooker(k , d , onlooker_courant, indice_choisi , population, Gbest,fitness_onlooker_courant, essais[indice_choisi], fitness)
                fitness_voisine = fit(data, k, voisine)
                
                if (fitness_onlooker_courant < fitness_voisine):
                    population[indice_choisi] = voisine
                    fitness[indice_choisi] = fitness_voisine
                    essais[indice_choisi] = 0
                    if (fitness_voisine > Fbest):
                        Fbest = fitness_voisine
                        Gbest = voisine
                else:
                    essais[indice_choisi] += 1

                           
            
            #Phase mutation
            for i in range(SN):
                S_new = mutation(k, d, solution, i, population, Gbest)
                F_new = fit(data, k, S_new)
                 
                if (fitness[i] < F_new):
                    population[i] = S_new
                    fitness[i] = F_new
                    essais[i] = 0
                    if (F_new > Fbest):
                        Fbest = F_new
                        Gbest = S_new
                else:
                    essais[i] += 1
                
            
            
            #Phase scout
            for i in range(SN):
                if (essais[i]>limit):
                    essais[i]=0
                    nouvelle_solution = generer_solution(k, d, Xmin, Xmax)
                    population[i] = nouvelle_solution
                    nouvelle_fitness = fit(data, k, nouvelle_solution)
                    fitness[i] = nouvelle_fitness
                    if (nouvelle_fitness > Fbest):
                        Gbest = nouvelle_solution
                        Fbest = nouvelle_fitness
                    
        return Gbest







#ABC_D_K -----------------------------------------------------------------


def ABC_DE_K(data, k):
        
        global nb_appel_obj
        
        nb_appel_obj = 0
        essais = np.zeros(SN, dtype=int)
        
        d=np.size(data[0])
        
        Xmin = np.empty(d)
        Xmax = np.empty(d)

        for i in range(d):
            Xmax[i] = np.amax(data.T[i])
            Xmin[i] = np.amin(data.T[i])
            
        population=np.empty((SN,k,d));
        fitness=np.empty((SN))

        Gbest = np.array(k_means.k_means(data.tolist(), k))

        Fbest = fit(data, k, Gbest)
        
        
        for i in range(SN):
            solution = generer_solution(k, d, Xmin, Xmax)
            fitness_solution = fit(data, k, solution)
            population[i] = solution
            fitness[i] = fitness_solution
        
            if (fitness_solution > Fbest):
                Gbest = solution
                Fbest = fitness_solution
            
        
        while (nb_appel_obj <= MAX_FE): 
            F1 = random.uniform(0,1)
            
            #phase des employĂŠes :
            for i, solution_courante in enumerate(population):
                    fitness_solution_courante = fitness[i]
                    
                    voisine = voisinage_employed(k , d , solution_courante , i, population, Gbest, fitness_solution_courante, essais[i],fitness,F1)
                    fitness_voisine = fit(data, k, voisine)
                    if (fitness_voisine > fitness_solution_courante):
                        population[i] = voisine
                        fitness[i] = fitness_voisine
                        essais[i] = 0
                        if (fitness_voisine > Fbest):
                            Gbest = voisine
                            Fbest = fitness_voisine
                        else:
                            essais[i] += 1
            
                    
            #calcul des probabilites
            proba = np.empty(SN)
            proba = calcul_proba(fitness)
        
            
            #phase des onlookers
            for i in range(SN):
                #selection d'une solution
                indice_choisi = choix_onlooker1(population,fitness,proba)
                onlooker_courant = population[indice_choisi]
                fitness_onlooker_courant = fitness[indice_choisi]
                
                #Exploitation de la solution
                voisine = voisinage_onlooker(k , d , onlooker_courant, indice_choisi , population, Gbest,fitness_onlooker_courant, essais[indice_choisi], fitness)
                fitness_voisine = fit(data, k, voisine)
                
                if (fitness_onlooker_courant < fitness_voisine):
                    population[indice_choisi] = voisine
                    fitness[indice_choisi] = fitness_voisine
                    essais[indice_choisi] = 0
                    if (fitness_voisine > Fbest):
                        Fbest = fitness_voisine
                        Gbest = voisine
                else:
                    essais[indice_choisi] += 1

                           
            #Phase mutation
            for i in range(SN):
                S_new = mutation(k, d, solution, i, population, Gbest)
                F_new = fit(data, k, S_new)
                 
                if (fitness[i] < F_new):
                    population[i] = S_new
                    fitness[i] = F_new
                    essais[i] = 0
                    if (F_new > Fbest):
                        Fbest = F_new
                        Gbest = S_new
                else:
                    essais[i] += 1
                
            
            
            #Phase scout
            for i in range(SN):
                if (essais[i]>limit):
                    essais[i]=0
                    nouvelle_solution = generer_solution(k, d, Xmin, Xmax)
                    population[i] = nouvelle_solution
                    nouvelle_fitness = fit(data, k, nouvelle_solution)
                    fitness[i] = nouvelle_fitness
                    if (nouvelle_fitness > Fbest):
                        Gbest = nouvelle_solution
                        Fbest = nouvelle_fitness
                    
        return Gbest



def DE_rand1_bin(data , k):
    
    #Génération de la population initiale
    global nb_appel_obj
    
    nb_appel_obj = 0
    
    d=np.size(data[0])
        
    Xmin = np.empty(d)
    Xmax = np.empty(d)

    for i in range(d):
        Xmax[i] = np.amax(data.T[i])
        Xmin[i] = np.amin(data.T[i])
    
    population=np.empty((pop_size,k,d));
    fitness=np.empty((pop_size))
    Gbest = generer_solution(k, d, Xmin, Xmax)
    Fbest = fit(data, k, Gbest)

    for i in range(pop_size):
        solution = generer_solution(k, d, Xmin, Xmax)
        fitness_solution = fit(data, k, solution)
        population[i] = solution
        fitness[i] = fitness_solution
        if (fitness_solution > Fbest):
            Gbest = solution
            Fbest = fitness_solution
    
    while(nb_appel_obj <= MAX_FE):
        for i in range(pop_size):
            fitness_parent = fitness[i]
            
            #génération du mutant
            trial_vector = rand1(k, d, i, population, Gbest, F)
            #génération du descendant
            enfant = binomial(k, d, trial_vector, population[i])
            fitness_enfant = fit(data, k , enfant)
            
            #sélection entre le parent et le descendant
            if(fitness_enfant > fitness_parent):
                population[i] = enfant
                fitness[i] = fitness_enfant
                if(fitness_enfant > Fbest):
                    Gbest = enfant
                    Fbest = fitness_enfant
        
    return Gbest



#current to rand ---------------------------------------------------

def DE_current_to_rand1(data , k):
    
    #generation de la population initiale
    global nb_appel_obj
    
    nb_appel_obj = 0
    
    d=np.size(data[0])
        
    Xmin = np.empty(d)
    Xmax = np.empty(d)

    for i in range(d):
        Xmax[i] = np.amax(data.T[i])
        Xmin[i] = np.amin(data.T[i])
    
    population=np.empty((pop_size,k,d));
    fitness=np.empty((pop_size))
    Gbest = generer_solution(k, d, Xmin, Xmax)
    Fbest = fit(data, k, Gbest)

    for i in range(pop_size):
        solution = generer_solution(k, d, Xmin, Xmax)
        fitness_solution = fit(data, k, solution)
        population[i] = solution
        fitness[i] = fitness_solution
        if (fitness_solution > Fbest):
            Gbest = solution
            Fbest = fitness_solution
    
    while(nb_appel_obj <= MAX_FE):
        F1 = random.uniform(0,1)
        for i in range(pop_size):
            fitness_parent = fitness[i]
            
            #generation du mutant 
            trial_vector = current_to_rand1(k, d, i, population, Gbest, F1)
            enfant = trial_vector
            fitness_enfant = fit(data, k , enfant)
            
            #sélection entre le mutant et le parent
            if(fitness_enfant > fitness_parent):
                population[i] = enfant
                fitness[i] = fitness_enfant
                if(fitness_enfant > Fbest):
                    Gbest = enfant
                    Fbest = fitness_enfant
        
    return Gbest