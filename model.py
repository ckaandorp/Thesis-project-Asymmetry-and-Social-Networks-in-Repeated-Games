import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
start_time = time.time()


class Model():
    """
    Model used to simulate agents learning of several generations.
    Returns the roster of all student objects.

    Attributes:
    n: number of agents
    roster: list of all agents (Student objects)
    asymmetry: boolean, if True model runs with asymmetric expansion
    social_network: boolean, if True model runs with social networks expansion
    update_size: number of strategies to delete each generation
    ranking: dictionary to keep track of avereage strategy scores and students
    avg_stamina: average stamina, either averaged from the asymmetric
                 distrbution or set at the start.
    avg_motivation: average motivation, either averaged from the asymmetric
                    distrbution or set at the start.
    connections: how many social groups each agent can be in.
    avg_gen_score: list of average score per generation.
    strat_dist: dictionary keeping track of the distribution of the strategies.

    """
    def __init__(self, n=1000, update_percentage=0.02, asymmetry=True,
                 social_network=True, homophily=True, motivation=0.5,
                 stamina=5, connections=2, n_groups=100, roster=None, name=''):
        self.n = n
        if roster:
            self.roster = roster
        else:
            self.roster = []
        self.asymmetry = asymmetry
        self.social_network = social_network
        self.update_size = round(update_percentage * n)
        self.ranking = {}
        self.similarity = 0.3

        # asymmetry variables
        self.avg_stamina = stamina
        self.avg_motivation = motivation

        # social network variables
        self.connections = connections
        self.n_groups = n_groups
        self.homophily = homophily

        # plot variables
        self.avg_gen_score = []
        self.strat_dist = {}
        self.name = name

    def __del__(self):
        """Disconnects student and social group objects"""
        for student in self.roster:
            for group in student.network:
                group.students.remove(student)
            del student

    def initialization(self):
        # base strategies to be evenly distributed
        base_strategies = [[0, 0, 0, 0, 0], [2, 0, 2, 0, 0], [2, 1, 2, 1, 1],
                           [1, 1, 1, 1, 1], [0, 1, 0, 1, 1]]

        # initialize student objects
        if self.asymmetry:
            # create random attributes
            stamina_dist = np.around(np.random.uniform(1, 10, self.n))
            motivation_dist = np.random.uniform(0, 1, self.n)
            # save average for symmetric simulations
            self.avg_stamina = np.mean(stamina_dist)
            self.avg_motivation = np.mean(motivation_dist)

            self.create_asymmetric_students(base_strategies, stamina_dist,
                                            motivation_dist)
        else:
            self.create_students(base_strategies)

        print("average stamina: ", self.avg_stamina, file=data)
        print("average motivation: ", self.avg_motivation, file=data)
        # prevent strategy order from impacting results
        random.shuffle(self.roster)

        if self.social_network:
            # make social networks
            self.social_grouper()

        return self.roster

    def social_grouper(self):
        """Initializes and fills the different social groups."""
        groups = []
        # create groups
        for _ in range(self.n_groups):
            a = Social_group()
            groups.append(a)

        if self.homophily:
            # search for best fitting groups per student
            for student in self.roster:
                self.find_group(student, groups)
        else:
            for student in self.roster:
                self.no_homo(student, groups)

    def find_group(self, student, groups):
        """
        Takes a student object and returns the groups which attributes
        are most similar to the students' attributes.
        student: Student object
        groups: list of group objects
        """
        stamina = student.stamina
        motivation = student.motivation
        diffs = []
        max_size = self.n / len(groups) * self.connections

        for i in range(len(groups)):
            # ignore groups that are full
            if groups[i].len == max_size:
                continue

            diff = self.difference_calc(stamina, motivation, groups[i].stamina,
                                        groups[i].motivation)

            # keep track of all differences per group (index)
            diffs.append((diff, i))

        diffs.sort()

        choices = [groups[diffs[x][1]] for x in range(self.connections)]

        self.friend_maker(student, choices)

    def no_homo(self, student, groups):
        """Add student to social group not based on homophily"""
        # sort by smallest group
        groups.sort(key=lambda x: x.len)
        choices = [groups[x] for x in range(self.connections)]
        self.friend_maker(student, choices)

    def friend_maker(self, student, choices):
        """
        Connects and updates student and group pairs
        student: Student object to be added to group
        choices: list of group objects that student will be added to
        """
        for group in choices:
            student.network.append(group)
            group.students.append(student)
            # caculate group average
            group.stamina = (group.len * group.stamina +
                             student.stamina) / (group.len + 1)
            group.motivation = (group.len * group.motivation +
                                student.motivation) / (group.len + 1)
            group.len += 1

    def difference_calc(self, s0, m0, s1, m1):
        """
        Returns total difference between student attributes and group
        attributes.
        s0: integer, student stamina
        m0: float, student motivation
        s1: integer, group stamina
        m1: float, group motivation
        """
        # if group average == 0, the group is empty and there should be no
        # diference between the group and student attributes
        if m1 == 0:
            return 0
        return abs(s0 - s1) + abs(m0 - m1) / 10

    def create_students(self, base_strategies):
        """
        Creates all students with uniform motivation and stamina and
        distributes the strategies among all students.
        base_strategies: list of stratgies.
        """
        strat_i = 0
        for i in range(self.n):
            a = Student(base_strategies[strat_i], self.avg_motivation,
                        self.avg_stamina)
            self.roster.append(a)
            # loop through the 5 strategies
            strat_i += 1
            if strat_i > 4:
                strat_i = 0

    def create_asymmetric_students(self, base_strategies, stamina_dist,
                                   motivation_dist):
        """
        Creates all students using generated motivation and stamina
        distributions and distributes the strategies among all students.
        base_strategies: list of stratgies.
        stamina_dist: randomly generated list of stamina values
        motivation_dist: randomly generated list of motivation values
        """
        strat_i = 0
        for i in range(self.n):
            a = Student(base_strategies[strat_i], motivation_dist[i],
                        stamina_dist[i])
            self.roster.append(a)
            strat_i += 1
            if strat_i > 4:
                strat_i = 0

    def simulate(self, k, plot=True):
        """
        Simulates k generations.
        """
        for i in range(k):
            self.ranking = {}
            # reset students' stats for the new week
            for student in self.roster:
                student.stamina = student.base_stamina
                student.motivation = student.base_motivation
                student.score = 0

            # simulate the week
            self.week()

            # replace strategies according to update size
            for j in range(self.update_size):

                worst_strategy = list(self.ranking.keys())[-1]

                # select student with worst strategy
                student = self.ranking[worst_strategy][2].pop()

                # Only immitate best strategy from peers if social groups, else
                # from entire population
                if self.social_network:
                    ranking = {}
                    for group in student.network:
                        for friend in group.students:
                            current_strat = friend.strategy
                            current_score = friend.score
                            try:
                                ranking[str(current_strat)][0] += 1
                                ranking[str(current_strat)][1] += current_score
                            except:
                                ranking[str(current_strat)] = [1, current_score]

                    # average score per strategy
                    for strat in ranking:
                        n_students = ranking[strat][0]  # number of students
                        average_score = ranking[strat][1] / n_students
                        ranking[str(strat)][1] = average_score

                    sorted_strats = {k: v for k, v in sorted(ranking.items(),
                                     key=lambda item: -item[1][1])}

                    best_strategy = list(sorted_strats.keys())[0]

                else:
                    best_strategy = list(self.ranking.keys())[0]

                # convert best_strategy back to list
                best_strategy = best_strategy.translate({ord(i): None for i in '[],'})
                best_strategy = [int(x) for x in best_strategy.split(" ")]

                student.strategy = best_strategy

                # delete strategy entry if no students use that strategy
                if not self.ranking[worst_strategy][2]:
                    del self.ranking[worst_strategy]

            # collect data on distribution of strategies for each generation
            for strat in self.ranking:
                # amount of students that use this strategy
                n = self.ranking[strat][0]
                # add amount of students to corresponding strategy list, or
                # create new strategy entry if strategy was unknown
                try:
                    self.strat_dist[str(strat)].append(n)
                except:
                    self.strat_dist[str(strat)] = [n]

        # overview of strategies to be outputted to data.txt
        print("_" * 20, "\n", self.name, "\n" + "_" * 20, file=data)
        strat_dict = {}
        for strat in self.ranking.keys():
            print(strat, file=data)
            n_students = self.ranking[strat][0]
            strat_dict[strat] = n_students
            print("number of students    :", n_students, file=data)
            print("average strategy score:", self.ranking[strat][1], file=data)
            students = self.ranking[strat][2]
            highest = max(student.score for student in students)
            lowest = min(student.score for student in students)
            difference = highest - lowest
            print("max score    :", highest, file=data)
            print("min score    :", lowest, file=data)
            print("difference   :", difference, "\n", file=data)
        print("Average score over the generations\n", self.avg_gen_score,
              "\n", file=data)

        return self.avg_gen_score, self.strat_dist

    def strategyPrinter(self):
        """
        Saves all strategies and occurences in descending order.
        Returns strategy dictionary containting per strategy:
        [0] number of students
        [1] average score
        [2] list of students
        """
        strategy_stats = {}

        # collect all stategies
        for i in range(self.n):
            student = self.roster[i]
            current_strat = student.strategy
            current_score = student.score

            # add to dictionary
            try:
                strategy_stats[str(current_strat)][0] += 1
                strategy_stats[str(current_strat)][1] += current_score
                strategy_stats[str(current_strat)][2] += [student]
            except:
                strategy_stats[str(current_strat)] = [1, current_score,
                                                      [student]]

        gen_avg = 0
        # average score per strategy
        for strat in strategy_stats:
            n_students = strategy_stats[strat][0]  # number of students
            gen_avg += strategy_stats[strat][1]  # total score of all students
            average_score = strategy_stats[strat][1] / n_students
            strategy_stats[str(strat)][1] = average_score

        # add total average score per generation
        self.avg_gen_score.append(gen_avg / self.n)

        # sort in descending order
        # - causes the reverse sort
        sorted_strats = {k: v for k, v in sorted(strategy_stats.items(),
                         key=lambda item: -item[1][1])}

        self.ranking = sorted_strats

    def week(self):
        "Update student scores after the workweek."
        # simulate 5 day workweek
        for i in range(5):
            # reset study group configurations
            study_groups = [Social_group()]
            # shuffle student order to prevent same study group configurations
            random.shuffle(self.roster)
            for student in self.roster:
                choice = student.strategy[i]
                if choice == 0:  # self study
                    self.score_calc(student, 25, -0.2, 1)
                elif choice == 1:  # study group
                    self.study_group_maker(study_groups, student)
                elif choice == 2:  # lecture
                    self.score_calc(student, 25, -0.1, 2)

            self.study_group_scorer(study_groups)
        # collect data
        self.strategyPrinter()

    def study_group_maker(self, study_groups, student):
        stamina = student.stamina
        motivation = student.motivation
        diffs = []

        # go through all available groups
        for i in range(len(study_groups)):
            diff = self.difference_calc(stamina, motivation,
                                        study_groups[i].stamina,
                                        study_groups[i].motivation)

            # keep track of all differences per group and save index
            diffs.append((diff, i))

        # find group with least difference
        diffs.sort()
        diff = diffs[0][0]
        group = study_groups[diffs[0][1]]

        #  join group if group is similar enough and join chance is passed,
        #  else create own group
        if diff <= self.similarity and \
           random.random() < group.study():
            # caculate group average
            group.stamina = (group.len * group.stamina +
                             student.stamina) / (group.len + 1)
            group.motivation = (group.len * group.motivation +
                                student.motivation) / (group.len + 1)
            group.students.append(student)
            group.len += 1
        else:
            a = Social_group()
            a.stamina = stamina
            a.motivation = motivation
            a.students.append(student)
            a.len = 1
            study_groups.append(a)

    def study_group_scorer(self, study_groups):
        """
        Goes through every study group to check size and assigns appropriate
        score to every student in said group.
        study_groups: dictionary containing all groups with students
        """
        for group in study_groups:
            base_score = self.group_equation(group.len)
            for student in group.students:
                # The people in the study group influence your performance
                req_motivation = 0.1 + ((group.motivation - student.motivation) / 2)
                req_stamina = 3 - ((group.stamina - student.stamina) / 2)
                self.score_calc(student, base_score, req_motivation, req_stamina)

    def group_equation(self, n):
        if n == 1:
            return 10
        elif n == 2:
            return 25
        elif n == 3:
            return 28
        elif n == 4:
            return 30
        elif n == 5:
            return 26
        elif n == 6:
            return 24
        elif n == 7:
            return 21
        else:
            return 19

    def score_calc(self, student, base_score, req_mot, req_stamina):
        """
        Calculates the achieved score for the student
        student: student object
        base_score: base score of the study activity as integer
        req_motivation: required motivation for the study activity as float
        req_stamina: required stamina for the study activity as integer
        """
        student.stamina -= req_stamina
        # halve motivation if stamina is depleted
        if student.stamina < 1:
            student.motivation /= 2
        motivation = student.motivation + req_mot
        if motivation < 0:
            motivation = 0
        student.score += base_score * motivation


class Student():
    def __init__(self, strategy, motivation, stamina, network=None):
        self.strategy = strategy
        self.base_motivation = motivation
        self.motivation = self.base_motivation
        self.base_stamina = stamina
        self.stamina = self.base_stamina
        # simply passing network as [] by default causes erratic behaviour
        if network:
            self.network = network
        else:
            self.network = []
        self.score = 0


class Social_group():
    def __init__(self):
        self.stamina = 0
        self.motivation = 0
        self.students = []
        self.len = 0

    def study(self):
        """Returns chance to join study group based on groupsize"""
        return 1 + (4 - self.len) * 2.5 / 10


def copy_students(roster, motivation=None, stamina=None):
    """Copies all student objects"""
    copied_roster = []
    # if an average motivation is passed, create symmetric students using the
    # passed motivation and stamina. Else perserve asymmetric attributes.
    if motivation:
        for student in roster:
            a = Student(student.strategy, motivation, stamina,
                        network=student.network)
            copied_roster.append(a)
    else:
        for student in roster:
            a = Student(student.strategy, student.motivation, student.stamina,
                        network=student.network)
            copied_roster.append(a)
    return copied_roster


def friend_maker_outside(students):
    """
    Reconnects copied students and existing group pairs
    student: Student object to be added to group
    """
    for student in students:
        choices = student.network
        for group in choices:
            group.students.append(student)


def run_model(n=1000, k=50, update_percentage=0.02, n_groups=100,
              homophily=True, connections=2):
    """Runs the four different versions of the model"""
    avg_gen_score = []
    strategies = {}
    if not os.path.exists("Graphs"):
        os.mkdir("Graphs")

    print("Initializing first model and students")
    # model with asymmetric agents and social networks
    model = Model(n=n, update_percentage=update_percentage, n_groups=n_groups,
                  homophily=homophily, connections=connections,
                  name="Asymmetry and Social Networks")
    roster = model.initialization()

    # save data for the next simulations
    avg_motivation, avg_stamina = model.avg_motivation, model.avg_stamina
    roster_asym = copy_students(roster)
    roster_soc = copy_students(roster, motivation=avg_motivation,
                               stamina=avg_stamina)

    print("Simulating Asymmetric and Social Network model")
    score, strat_dict = model.simulate(k)
    avg_gen_score.append(score)
    AS_score.append(score)
    AS_strategies.append(strat_dict)
    del model

    print("Finished simulating first model, starting simulation of Asymmetric model")
    # asymmetric only model
    asym_model = Model(n=n, update_percentage=update_percentage,
                       homophily=homophily, n_groups=n_groups,
                       roster=roster_asym, social_network=False,
                       connections=connections, name="Asymmetry")
    score, strat_dict = asym_model.simulate(k)
    avg_gen_score.append(score)
    A_score.append(score)
    A_strategies.append(strat_dict)
    del asym_model

    print("Finished simulating Asymmetric model, starting simulation of Social Network model")
    # social nework only model
    friend_maker_outside(roster_soc)
    soc_model = Model(n=n, update_percentage=update_percentage,
                      homophily=homophily, n_groups=n_groups,
                      roster=roster_soc, asymmetry=False,
                      connections=connections, name="Social Network")
    score, strat_dict = soc_model.simulate(k)
    avg_gen_score.append(score)
    S_score.append(score)
    S_strategies.append(strat_dict)
    del soc_model

    print("Finished simulating Social Network model, starting simulation of base model")
    # neither
    base_model = Model(n=n, update_percentage=update_percentage,
                       homophily=homophily, n_groups=n_groups,
                       motivation=avg_motivation, stamina=avg_stamina,
                       asymmetry=False, social_network=False, name="Base")
    base_model.initialization()
    score, strat_dict = base_model.simulate(k)
    avg_gen_score.append(score)
    B_score.append(score)
    B_strategies.append(strat_dict)

    print("\nStrategy occurences: ", strategies, "\n\n", file=data)
    print("Finished all simulations")


def run(times=20, generations=100, homophily=True, connections=2):
    """Runs the model multiple times to allow for an average to be calculated"""
    for i in range(times):
        print("\n---- Simulation: ", i, "----")
        print("\n---- Simulation: ", i, "----", file=data)
        run_model(k=generations, homophily=homophily, connections=connections)


def plot_average_score():
    # zips average score per generation together to allow for averageging
    total_AS_score = list(zip(*AS_score))
    total_A_score = list(zip(*A_score))
    total_S_score = list(zip(*S_score))
    total_B_score = list(zip(*B_score))

    total_AS = []
    total_A = []
    total_S = []
    total_B = []

    # calculate average over all simulations
    for i in range(len(total_AS_score)):
        total_AS.append(np.mean(total_AS_score[i]))
        total_A.append(np.mean(total_A_score[i]))
        total_S.append(np.mean(total_S_score[i]))
        total_B.append(np.mean(total_B_score[i]))

    all_averages = [total_AS, total_A, total_S, total_B]

    # plot average score per version
    for i in range(len(all_averages)):
        print("\nAverage score " + titles[i] + ":", all_averages[i], file=data)
        title = "Average score per generation for " + titles[i] + " model"
        plt.plot(all_averages[i])
        plt.title(title)
        plt.xlabel("Generations")
        plt.ylabel("Average score")
        plt.savefig("Graphs/" + title)
        plt.close()

    # plot all versions together
    for i in range(len(all_averages)):
        plt.plot(all_averages[i], label=titles[i])
    plt.title("Average score per generation")
    plt.xlabel("Generations")
    plt.ylabel("Average score")
    plt.legend()
    plt.savefig("Graphs/Average score over generations")
    plt.close()


def plot_strategy_dists(generations):
    # list of strategy distribution dictionaries per version
    all_strategies_dists = [AS_strategies, A_strategies, S_strategies,
                            B_strategies]

    for i in range(len(all_strategies_dists)):
        # dictionary that will contain the average strategy progress
        end_dict = {}
        # convert to dataframe for easier acces
        df = pd.DataFrame(all_strategies_dists[i])
        # calculate average progress of each strategy
        for strat in df:
            new_progress = []
            all_simulations = df[strat].tolist()
            for single_sim in all_simulations:
                # pad with zeroes if strategy didnt exist at timestep
                single_sim += [0] * (generations - len(single_sim))
            combi = list(zip(*all_simulations))
            for all_gens in combi:
                new_progress.append(np.mean(all_gens))
            end_dict[strat] = new_progress

        print("\nAverage strategy dist of " + titles[i] + ":", end_dict, file=data)
        # plot progression
        title = "Strategy distribution for " + titles[i] + " model"
        plt.title(title)
        plt.xlabel("Generations")
        plt.ylabel("Amount of students")
        for strat in end_dict:
            plt.plot(end_dict[strat], label=strat, color=colors[strat])
        plt.legend()
        plt.savefig("Graphs/" + title)
        plt.close()


data = open("data.txt", "w")

# collectors of all average scorings over the 20 simulations
AS_score = []
A_score = []
S_score = []
B_score = []

AS_strategies = []
A_strategies = []
S_strategies = []
B_strategies = []

# versions of the model
titles = {0: "Asymmetric and Social Network",
          1: "Asymmetric",
          2: "Social Network",
          3: "Base"}

# colors used for plotting the different strategies
colors = {'[0, 0, 0, 0, 0]': 'blue',
          '[2, 0, 2, 0, 0]': 'orange',
          '[2, 1, 2, 1, 1]': 'purple',
          '[1, 1, 1, 1, 1]': 'red',
          '[0, 1, 0, 1, 1]': 'green'}

# how many generations/weeks will be simulated
generations = 100

run(times=20, generations=generations)

plot_average_score()
plot_strategy_dists(generations)

data.close()
print("--- %s seconds ---" % (time.time() - start_time))
