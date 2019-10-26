import numpy as np
#N multiplo de 2 y N_archive
def SPEA2(N,N_archive,T,n_genes, objective_function, discrete = False, interval = (0,1) ,maximize=False):
    #Falta evaluar la funcion objetivo
    #Cambiar el nombre de objective_values por algo del espacio objetivo
    #Tratar los individuos no validos
    #Generar una poblacion inicial
    k = int(np.sqrt(N+N_archive))
    population = None
    total_population = None
    archive = None
    objective_values = None
    t = 0 #Generation 
    k_distance = None
    #np.random.seed(seed=1000)

    #Generar poblacion inicial
    if not discrete:
        total_population = np.random.uniform(interval[0],interval[1],(N+N_archive,n_genes))
    else:
        total_population = np.random.random_integers(interval[0],interval[1],(N+N_archive,n_genes))

    #total_population = np.round(total_population, 3)
       

    while True:
        np.random.shuffle(total_population)
        total_population = np.round(total_population, 3)
        #Evaluar poblacion
        print(total_population)
        objective_values = objective_function(total_population)
        objective_values = np.round(objective_values, 3)

        print(objective_values)
        print("poblacion y obj")
        #input()

        #Calcula S = numero de gente que dominas si maximizar la funcion
        equals = np.equal(objective_values[:,np.newaxis,:],objective_values[np.newaxis,:,:])
        if maximize:
            dif = objective_values[:,np.newaxis,:]<objective_values[np.newaxis,:,:]
        else:
            #Calcula S = numero de gente que dominas si minimizar la funcion
            dif = objective_values[:,np.newaxis,:]>=objective_values[np.newaxis,:,:]

        dif = np.count_nonzero(dif,axis=2)
        #np.fill_diagonal(dif,0)
        dif = dif// objective_values.shape[1] #Matriz con la gente dominada

        equals = np.count_nonzero(equals,axis=2)
        #np.fill_diagonal(equals,0)
        equals = equals// objective_values.shape[1] #Matriz con la gente dominada
        equals = 1 - equals

        dominated = np.logical_and(dif,equals)


        S = dominated.sum(axis=0)
        
        #calcula R = sumatorio del numero de gente que domina la gente que me domina
        R = S*dominated
        R = R.sum(axis=1)
        print(dominated)
        print(S)
        print(R)
        
        #Calcular distancias
        #distances = np.sqrt((objective_values[:,np.newaxis,:]-objective_values[np.newaxis,:,:])**2)
        distances = (objective_values[:,np.newaxis,:]-objective_values[np.newaxis,:,:])**2
        distances = distances.sum(axis=2)
        distances = np.sqrt(distances)
        ind = np.triu_indices(distances.shape[0])
        #La zona triangular superior se pone a valores altos para no tenerla en cuenta
        np.fill_diagonal(distances,10000)
    
        distances = np.sort(distances, axis=1)

        #Se seleccionan los indices del eje y  de las diferencias de las distancias. Indica un inidividuo
        k_distance = distances[:,k]

        D = 1/(k_distance+2)
   
        #Fitness
        F = D+R
        print("Fitnes ",F)
        #input()


        #Environmental Selection
        del archive
        indices_next_archive = np.where(F<1)[0]
        print(indices_next_archive)
        print(objective_values[indices_next_archive])
        length = len(indices_next_archive)
        if  length < N_archive:
            rest = N_archive - length
            best_inidices = np.argsort(F)
            indices_next_archive = best_inidices[:N_archive]
        elif length > N_archive:
            rest = length - N_archive

            distances = np.sqrt((objective_values[[indices_next_archive],np.newaxis,:]-objective_values[np.newaxis,[indices_next_archive],:])**2)
            distances = distances.sum(axis=2)
            dist = (objective_values[[indices_next_archive],np.newaxis,:]-objective_values[np.newaxis,[indices_next_archive],:])**2
            dist = np.squeeze(dist)
            dist = np.sum(dist, axis=2)
            dist = np.sqrt(dist)


            #Seleccionar las distancias k de los indices que estan en el archive
            #k_distance_archive = k_distance[indices_next_archive]
            #Hacer la resta de todas las distancias contodas y abs
            #dif_k_dist = np.abs(k_distance_archive[np.newaxis,:] - k_distance_archive[:,np.newaxis])
            #Se obtienen los indices de una matriz triangular superior de tamaño dif_k_dist
            #ind = np.triu_indices(dif_k_dist.shape[0])
            #La zona triangular superior se pone a valores altos para no tenerla en cuenta
            #dif_k_dist[ind] = 10000
            ind = np.triu_indices(dist.shape[0])
            dist[ind] = 10000
            #Se seleccionan los indices del eje y  de las diferencias de las distancias. Indica un inidividuo
            #y,_ = np.unravel_index(np.argsort(dif_k_dist.ravel()), dif_k_dist.shape)
            y,_ = np.unravel_index(np.argsort(dist.ravel()), dist.shape)
            #Seleccionar el indice de los indices que aparece por primera vez ordenados de valor mas pequeño a mas alto.
            _ , i = np.unique(y, return_index=True)
            #Los indices de los individuos en el archive que no iran al archive ordenados por preferencia.
            indices_to_delete = y[sorted(i)][:rest]
            #indices_next_archive = indices_next_archive[indices_to_delete]#se quitan los indices que no valen.
            indices_next_archive = np.delete(indices_next_archive, indices_to_delete)
            print(dist)
            print(indices_next_archive)
            #input()

        #print("archive ",indices_next_archive)


        archive = total_population[indices_next_archive]
        archive_F = F[indices_next_archive]
        del objective_values

        #STOP CONDITION
        if t >= T:
            return archive

        #Tournament
        rand_indices = np.random.randint(archive_F.shape[0],size=(2,N))#Parejas de numeros aleatorios
        match = np.array([archive_F[rand_indices[0]],archive_F[rand_indices[1]]])#Selecciona que individuos se enfrentan
        winers = match.min(axis=0)#Coge el fitness con menor valor
        index_winers = np.searchsorted(archive_F,winers)#Coge el indice de los individuos ganadores para el meeting pool
        _, index_winers = np.where(np.equal(archive_F[np.newaxis,:],winers[:,np.newaxis]))

        meeting_pool = archive[index_winers]
        #print(indices_next_archive)
        #print("POOL",index_winers)


        if not discrete:
            #Cruce morfologico
            parents = []

            #Append de los padres por algun motivo
            for i in range(N//2):
                #Cogen 5 padres de forma aleatoria
                parent_index = np.random.choice(len(meeting_pool), 5, replace=False)
                #Se añaden los padres
                parents.append(meeting_pool[np.newaxis,parent_index])

            #Fin del bucle de seleccion de cruce

            limits = np.concatenate(parents)#se juntan los padres de cada descendiente, que van a indicar los limites de cada gen

            #Hay que normalizar el limite
            #limits 

            #print(limits)

            maximum = np.max(limits,axis=1)
            minimum = np.min(limits,axis=1)

            #print(maximum)
            #print(minimum)
            G = maximum - minimum #calcula g_i 

            lt = np.where(G<=0.54)
            gt = np.where(G>0.54)
            G[lt] = -(0.25 * G[lt])-0.001
            G[gt] = (0.5 * G[gt])-0.265
            #Ahora G es fi(gi)
            fi_G = G

            G_max = maximum - fi_G
            G_min = minimum + fi_G
            exceed_minimum = np.where(G_min<interval[0]) 
            exceed_maximum = np.where(G_max>interval[1]) 
            G_max[exceed_maximum] = interval[1]
            G_min[exceed_minimum] = interval[0]

    
            #print("MAX ",G_max)
            #print("MIN ", G_min)


            
            #Se seleccionan valores aleatorios en el intervalo definido para cada gen y cada hijo
            descendants = np.random.uniform(G_min.flatten(),G_max.flatten(),maximum.shape[0]*maximum.shape[1])
            #descendants = np.random.uniform(G_min.flatten(),G_max.flatten(),(maximum.shape[0],maximum.shape[1]))
            descendants.shape = maximum.shape #Se recupera la forma
            #exceed_minimum = np.where(descendants<interval[0]) 
            #exceed_maximum = np.where(descendants>interval[1]) 
            #descendants[exceed_minimum] = interval[0]
            #descendants[exceed_maximum] = interval[1]

            #second_descendants = np.random.uniform(G_min.flatten(),G_max.flatten(),maximum.shape[0]*maximum.shape[1])
            #second_descendants.shape = maximum.shape #Se recupera la forma
            second_descendants = (minimum + maximum) - descendants
            exceed_minimum = np.where(second_descendants<interval[0]) 
            rxceed_maximum = np.where(second_descendants>interval[1]) 
            second_descendants[exceed_minimum] = interval[0]
            second_descendants[exceed_maximum] = interval[1]

            #Fin cruce morfologico
        else:
            parents = []

            #Append de los padres por algun motivo
            n_parents = N//2
            for i in range(n_parents):
                #Cogen 2 progenitores de forma aleatoria
                parent_index = np.random.choice(len(meeting_pool), 2, replace=False)
                #Se añaden los progenitores
                parents.append(meeting_pool[np.newaxis,parent_index])

            parents =  np.concatenate(parents)
            descendants_masc = np.random.randint(interval[0],interval[1]+1, size=(n_parents,n_genes))
            descendants = parents[:,0] * descendants_masc + parents[:,1] * (1-descendants_masc)
            second_descendants = parents[:,0] * (1-descendants_masc) + parents[:,1] * descendants_masc


            
        del population
        population = np.concatenate((descendants,second_descendants))

        #Mutaciones
        mutation_p = np.random.rand(population.shape[0],population.shape[1])
        indices = np.where(mutation_p<=0.01)
        new_values_length = len(indices[0])
        if not discrete:
            population[indices] = np.random.rand(new_values_length)
        else:
            population[indices] = np.random.randint(interval[0],interval[1]+1,new_values_length)

        del total_population
        total_population = np.concatenate((population, archive))

        t += 1

        #input()





def normalize(matrix, interval):
    matrix = (matrix-interval[0])/(interval[1] - interval[0])
    return matrix

def restore(matrix, interval):
    matrix = matrix*(interval[1] - interval[0]) + interval[0]
    return matrix

def SPEA2SPECIAL(N,N_archive,T,n_genes, objective_function):
    #Falta evaluar la funcion objetivo
    #Cambiar el nombre de objective_values por algo del espacio objetivo
    #Tratar los individuos no validos
    #Generar una poblacion inicial
    k = int(np.sqrt(N+N_archive))
    population = None
    total_population = None
    archive = None
    objective_values = None
    t = 0 #Generation 
    k_distance = None
    interval_N = (0,1)
    interval = (-5,5)

    #np.random.seed(seed=1000)

    #Generar poblacion inicial
    total_population = np.random.random_integers(interval[0],interval[1],(N+N_archive,n_genes))
    total_population[:,0] = normalize(total_population[:,0],interval)

    #total_population = np.round(total_population, 3)
       

    while True:
        np.random.shuffle(total_population)
        total_population = np.round(total_population, 3)
        #Evaluar poblacion
        print(total_population)
        objective_values = objective_function(total_population)
        objective_values = np.round(objective_values, 3)

        print(objective_values)
        print("poblacion y obj")
        #input()

        #Calcula S = numero de gente que dominas si maximizar la funcion
        equals = np.equal(objective_values[:,np.newaxis,:],objective_values[np.newaxis,:,:])
        #Calcula S = numero de gente que dominas si minimizar la funcion
        dif = objective_values[:,np.newaxis,:]>=objective_values[np.newaxis,:,:]

        dif = np.count_nonzero(dif,axis=2)
        #np.fill_diagonal(dif,0)
        dif = dif// objective_values.shape[1] #Matriz con la gente dominada

        equals = np.count_nonzero(equals,axis=2)
        #np.fill_diagonal(equals,0)
        equals = equals// objective_values.shape[1] #Matriz con la gente dominada
        equals = 1 - equals

        dominated = np.logical_and(dif,equals)


        S = dominated.sum(axis=0)
        
        #calcula R = sumatorio del numero de gente que domina la gente que me domina
        R = S*dominated
        R = R.sum(axis=1)
        
        #Calcular distancias
        #distances = np.sqrt((objective_values[:,np.newaxis,:]-objective_values[np.newaxis,:,:])**2)
        distances = (objective_values[:,np.newaxis,:]-objective_values[np.newaxis,:,:])**2
        distances = distances.sum(axis=2)
        distances = np.sqrt(distances)
        ind = np.triu_indices(distances.shape[0])
        #La zona triangular superior se pone a valores altos para no tenerla en cuenta
        np.fill_diagonal(distances,10000)
    
        distances = np.sort(distances, axis=1)

        #Se seleccionan los indices del eje y  de las diferencias de las distancias. Indica un inidividuo
        k_distance = distances[:,k]

        D = 1/(k_distance+2)
   
        #Fitness
        F = D+R
        print("Fitnes ",F)
        #input()


        #Environmental Selection
        del archive
        indices_next_archive = np.where(F<1)[0]
        length = len(indices_next_archive)
        if  length < N_archive:
            rest = N_archive - length
            best_inidices = np.argsort(F)
            indices_next_archive = best_inidices[:N_archive]
        elif length > N_archive:
            rest = length - N_archive

            distances = np.sqrt((objective_values[[indices_next_archive],np.newaxis,:]-objective_values[np.newaxis,[indices_next_archive],:])**2)
            distances = distances.sum(axis=2)
            dist = (objective_values[[indices_next_archive],np.newaxis,:]-objective_values[np.newaxis,[indices_next_archive],:])**2
            dist = np.squeeze(dist)
            dist = np.sum(dist, axis=2)
            dist = np.sqrt(dist)


            #Seleccionar las distancias k de los indices que estan en el archive
            #k_distance_archive = k_distance[indices_next_archive]
            #Hacer la resta de todas las distancias contodas y abs
            #dif_k_dist = np.abs(k_distance_archive[np.newaxis,:] - k_distance_archive[:,np.newaxis])
            #Se obtienen los indices de una matriz triangular superior de tamaño dif_k_dist
            #ind = np.triu_indices(dif_k_dist.shape[0])
            #La zona triangular superior se pone a valores altos para no tenerla en cuenta
            #dif_k_dist[ind] = 10000
            ind = np.triu_indices(dist.shape[0])
            dist[ind] = 10000
            #Se seleccionan los indices del eje y  de las diferencias de las distancias. Indica un inidividuo
            #y,_ = np.unravel_index(np.argsort(dif_k_dist.ravel()), dif_k_dist.shape)
            y,_ = np.unravel_index(np.argsort(dist.ravel()), dist.shape)
            #Seleccionar el indice de los indices que aparece por primera vez ordenados de valor mas pequeño a mas alto.
            _ , i = np.unique(y, return_index=True)
            #Los indices de los individuos en el archive que no iran al archive ordenados por preferencia.
            indices_to_delete = y[sorted(i)][:rest]
            #indices_next_archive = indices_next_archive[indices_to_delete]#se quitan los indices que no valen.
            indices_next_archive = np.delete(indices_next_archive, indices_to_delete)
            #input()

        #print("archive ",indices_next_archive)


        archive = total_population[indices_next_archive]
        archive_F = F[indices_next_archive]
        del objective_values

        #STOP CONDITION
        if t >= T:
            return archive

        #Tournament
        rand_indices = np.random.randint(archive_F.shape[0],size=(2,N))#Parejas de numeros aleatorios
        match = np.array([archive_F[rand_indices[0]],archive_F[rand_indices[1]]])#Selecciona que individuos se enfrentan
        winers = match.min(axis=0)#Coge el fitness con menor valor
        index_winers = np.searchsorted(archive_F,winers)#Coge el indice de los individuos ganadores para el meeting pool
        _, index_winers = np.where(np.equal(archive_F[np.newaxis,:],winers[:,np.newaxis]))

        meeting_pool = archive[index_winers]
        #print(indices_next_archive)
        #print("POOL",index_winers)


        #Cruce morfologico
        parents = []

        #Append de los padres por algun motivo
        for i in range(N//2):
            #Cogen 5 padres de forma aleatoria
            parent_index = np.random.choice(len(meeting_pool), 5, replace=False)
            #Se añaden los padres
            parents.append(meeting_pool[np.newaxis,parent_index])

        #Fin del bucle de seleccion de cruce

        limits = np.concatenate(parents)#se juntan los padres de cada descendiente, que van a indicar los limites de cada gen
        #print("LIMITES SN",limits)
        #print("LO QUE PASO", limits[:,:, 1:])

        #Hay que normalizar el limite
        #limits[:,:, 1:] = normalize(limits[:,:, 1:], interval) 

        #print("LIMITES N",limits)

        maximum = np.max(limits,axis=1)
        minimum = np.min(limits,axis=1)
        #print("MAXimo ",maximum)
        #print("MINimo ", minimum)

        #print(maximum)
        #print(minimum)
        G = maximum - minimum #calcula g_i 

        lt = np.where(G<=0.54)
        gt = np.where(G>0.54)
        G[lt] = -(0.25 * G[lt])-0.001
        G[gt] = (0.5 * G[gt])-0.265
        #Ahora G es fi(gi)
        fi_G = G

        G_max = maximum - fi_G
        G_min = minimum + fi_G
        exceed_minimum = np.where(G_min[:,0]<interval_N[0]) 
        exceed_maximum = np.where(G_max[:,0]>interval_N[1]) 
        G_max[exceed_maximum] = interval_N[1]
        G_min[exceed_minimum] = interval_N[0]

        exceed_minimum = np.where(G_min[:,1:]<interval[0]) 
        exceed_maximum = np.where(G_max[:,1:]>interval[1]) 
        G_max[exceed_maximum] = interval[1]
        G_min[exceed_minimum] = interval[0]



    
        #print("MAX ",G_max)
        #print("MIN ", G_min)


        
        #Se seleccionan valores aleatorios en el intervalo definido para cada gen y cada hijo
        descendants = np.random.uniform(G_min.flatten(),G_max.flatten(),maximum.shape[0]*maximum.shape[1])
        #descendants = np.random.uniform(G_min.flatten(),G_max.flatten(),(maximum.shape[0],maximum.shape[1]))
        descendants.shape = maximum.shape #Se recupera la forma
        #exceed_minimum = np.where(descendants<interval[0]) 
        #exceed_maximum = np.where(descendants>interval[1]) 
        #descendants[exceed_minimum] = interval[0]
        #descendants[exceed_maximum] = interval[1]

        #second_descendants = np.random.uniform(G_min.flatten(),G_max.flatten(),maximum.shape[0]*maximum.shape[1])
        #second_descendants.shape = maximum.shape #Se recupera la forma
        second_descendants = (minimum + maximum) - descendants
        exceed_minimum = np.where(second_descendants[:,0]<interval_N[0]) 
        rxceed_maximum = np.where(second_descendants[:,0]>interval_N[1]) 
        second_descendants[exceed_minimum] = interval_N[0]
        second_descendants[exceed_maximum] = interval_N[1]

        exceed_minimum = np.where(second_descendants[:,1:]<interval[0]) 
        rxceed_maximum = np.where(second_descendants[:,1:]>interval[1]) 
        second_descendants[exceed_minimum] = interval[0]
        second_descendants[exceed_maximum] = interval[1]

        #Fin cruce morfologico
        
        del population
        population = np.concatenate((descendants,second_descendants))
        #print("HIJOS")
        #print(population)
        #population[:, 1:] = restore(population[:,1:], interval)
        #print(population)
        #print("FIN")

        #Mutaciones
        #mutation_p = np.random.rand(population.shape[0],population.shape[1])
        #indices = np.where(mutation_p<=0.01)
        #new_values_length = len(indices[0])
        #mutation = np.random.rand(population.shape[0],population.shape[1])
        #mutation[:, 1:] = restore(mutation[:,1:], interval)
        #print("MUTACIONES",mutation)
        #population[indices] = mutation[indices]

        del total_population
        total_population = np.concatenate((population, archive))

        t += 1

        #input()

