"""
### Nations

Each nation has a convex combination over 3 mentalities:
1. Builder - favors increasing efficiency of harvesting resources
2. Hoarder - hoards resources and items
3. Trader - builds surprluses through intelligent trade

Each nation has three parameters associated with it:
1. Aggression - How much stronger it should be compared to its weakest neighbor before it attacks that neighbor.
2. Caution - At what percentage of the nation's attack strength its strongest neighbor must be before the
nation starts hoarding exclusively.
3. Efficiency - Percentage of max resources that are harvested by the harvesting population of this
nation.
4. Fertility - how fast the population of this nation grows tick by tick.

Each nation is instantaited using two Dirichlet distributions:
1. Distribution over mentalities - np.random.default_rng().dirichlet([m_b, m_h, m_t])
2. Distribution over parameters - np.random.default_rng().dirichlet([p_a, p_c, p_e, p_f])

When a nation is created, it receives a certain starting population (provided as an argument at creation
time). It also receives stockpiles of each resource as the proportions of its allocated starting resources
corresponding to its parameter scores.

### Resources

Abstracted into 4 types of resources:
1. Offensive
2. Defensive
3. Exploitative
4. Procreative

Offensive resources aid in attacks. Defensive resources aid in defense. Exploitative resources improve
harvesting efficiency. Procreative resources are required to support population growth.

Each additional population created during a growth tick consumes a surplus procreative resource. Even
if a nation has high fertility, if it does not stockpile enough procreative resources, it will fail to
support any additional population in excess of its procreative resource stockpile.
"""
from dataclasses import dataclass
import json
import random
from typing import cast, List, Optional, Tuple

import numpy as np

# Resource order: offensive, defensive, exploitative, procreative
ResourceAmounts = List[int]


@dataclass
class Nation:
    # builder, hoarder, trader >= 0 and builder + hoarder + trader = 1
    builder: float
    hoarder: float
    trader: float

    # aggression, caution, efficiency, fertility >= 0 and aggression + caution + efficiency + fertility = 1
    aggression: float
    caution: float
    efficiency: float
    fertility: float

    # Current population of nation
    population: int

    # Current stocks of each resource
    stocks: ResourceAmounts

    # Current bids for each resource in terms of each other resource
    asks: List[ResourceAmounts]

    # Technology level determines efficiency as a logarithm of the technology level. Diminishing marginal
    # utility, but unbounded potential.
    # Cost to increase technology to level n is n exploitative resources. This makes the total cost of
    # getting to a given level of technology quadratic in the level. 1 + 2 + 3 + ... + n = n(n+1)/2.
    technology: int = 1


def generate_variants(
    mentality_alpha: Tuple[float, float, float],
    parameter_alpha: Tuple[float, float, float, float],
    num_variants: int,
    starting_population: int,
    starting_resources: int,
    starting_technology: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Nation]:
    """
    Generates a list of nations that are similar to each other in terms of their mentalities and parameters.
    """
    assert len(mentality_alpha) == 3, "mentality_alpha must be a tuple of length 3"
    assert len(parameter_alpha) == 4, "parameter_alpha must be a tuple of length 4"

    if rng is None:
        rng = np.random.default_rng()

    mentalities = rng.dirichlet(mentality_alpha, num_variants)
    parameters = rng.dirichlet(parameter_alpha, num_variants)

    nations: List[Nation] = []
    for mentality, params in zip(mentalities, parameters):
        builder, hoarder, trader = mentality
        aggression, caution, efficiency, fertility = params
        nation = Nation(
            builder=builder,
            hoarder=hoarder,
            trader=trader,
            aggression=aggression,
            caution=caution,
            efficiency=efficiency,
            fertility=fertility,
            population=starting_population,
            stocks=[
                int(starting_resources * aggression),
                int(starting_resources * caution),
                int(starting_resources * efficiency),
                int(starting_resources * fertility),
            ],
            asks=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            technology=starting_technology,
        )
        nations.append(nation)

    return nations


def generate_nations(
    mentality_alphas: List[Tuple[float, float, float]],
    parameter_alphas: List[Tuple[float, float, float, float]],
    starting_populations: List[int],
    starting_resources: List[int],
    starting_technologies: List[int],
    rng: Optional[np.random.Generator] = None,
) -> List[Nation]:
    assert len(mentality_alphas) == len(
        parameter_alphas
    ), "mentality_alphas and parameter_alphas must be the same length"
    assert len(mentality_alphas) == len(
        starting_populations
    ), "mentality_alphas and starting_populations must be the same length"
    assert len(mentality_alphas) == len(
        starting_resources
    ), "mentality_alphas and starting_resources must be the same length"

    if rng is None:
        rng = np.random.default_rng()

    nation_variants: List[List[Nation]] = [
        generate_variants(
            mentality_alpha,
            parameter_alpha,
            starting_population,
            starting_stockpile,
            starting_technology,
            rng,
        )
        for mentality_alpha, parameter_alpha, starting_population, starting_stockpile, starting_technology in zip(
            mentality_alphas,
            parameter_alphas,
            starting_populations,
            starting_resources,
            starting_technologies,
        )
    ]

    nations = [variants[0] for variants in nation_variants]
    return nations


class GameSession:
    """
    Implements game mechanics for a list of participating nations.
    """

    def __init__(
        self,
        nations: List[Nation],
        population_growth_interval: int = 10,
    ) -> None:
        """
        Initializes a game session with a list of nations and a population growth interval.

        The population growth interval specifies how many game ticks should pass between each population growth
        tick.
        """
        self.nations: List[Nation] = nations
        self.population_growth_interval: int = population_growth_interval
        self.last_tick: int = 0
        self.total_supply: ResourceAmounts = (0, 0, 0, 0)

        self.update()

    def update_total_supply(self) -> None:
        """
        Updates the total supply of each resource.
        """
        national_stocks_by_resource = zip(*[nation.stocks for nation in self.nations])
        self.total_supply = cast(
            ResourceAmounts,
            [sum(stocks) for stocks in national_stocks_by_resource],
        )

    def update_bids(self) -> None:
        """
        Updates the bids of each nation.

        Each bid is updated based on:
        1. The nation's current supply of each resource
        2. The nation's parameters (implies relative priority of resources)
        3. The global supply of each resource

        Suppose that d is the resource that a nation needs and we are calculating its bid for d in terms
        of s, a different resource that it has.

        Let p_d and p_s represent the parameters corresponding to d and s respectively.
        Let N_d and N_s represent the nation's supply of d and s respectively.
        Let G_d and G_s represent the global supply of d and s respectively.

        The the nation's bid for d in units of s is given by the following algorithm:
        1. If N_d/N_s >= p_d/p_s, then bids[d][s] = 0
        2. Else bids[d][s] = ceiling(sqrt((p_d/p_s) * (G_d/G_s))), the ceiling of the geometric mean
        of the nation's intrinsic price and the implied global price.
        """
        nation_parameters = [
            [
                nation.aggression,
                nation.caution,
                nation.efficiency,
                nation.fertility,
            ]
            for nation in self.nations
        ]
        national_preference_prices = np.array(
            [
                [
                    [
                        0
                        if s == d
                        else nation_parameters[i][s] / nation_parameters[i][d]
                        for s in range(4)
                    ]
                    for d in range(4)
                ]
                for i, nation in enumerate(self.nations)
            ]
        )

        national_supply_proportions = np.array(
            [
                [
                    [
                        0
                        if nation.stocks[d] == 0
                        else nation.stocks[s] / nation.stocks[d]
                        for s in range(4)
                    ]
                    for d in range(4)
                ]
                for nation in self.nations
            ]
        )

        global_supply_proportions = np.stack(
            [
                [
                    [
                        0
                        if self.total_supply[d] == 0
                        else self.total_supply[s] / self.total_supply[d]
                        for s in range(4)
                    ]
                    for d in range(4)
                ]
                for _ in self.nations
            ],
            axis=0,
        )

        vectorized_bids = np.ceil(
            np.sqrt(
                np.multiply(
                    np.multiply(national_preference_prices, global_supply_proportions),
                    national_supply_proportions < national_preference_prices,
                )
            )
        )

        for i, nation in enumerate(self.nations):
            nation.asks = vectorized_bids[i]

    def update(self) -> None:
        """
        Updates the game state.
        """
        self.update_total_supply()
        self.update_bids()

    def should_attack(self, nation_index: int) -> Optional[int]:
        """
        Maps aggression which is between 0 and 1 to the interval [2, 1]. That is the relative strength
        in offense that the nation must have over its weakest neighbor's defense before it mounts an attack.

        If it does have this strength, this function returns the index of the weakest neighbor. Otherwise,
        it returns None.
        """
        required_advantage = 2 - self.nations[nation_index].aggression

        offensive_strength = self.nations[nation_index].stocks[0]
        defensive_strengths = [
            (i, nation.stocks[1])
            for i, nation in enumerate(self.nations)
            if i != nation_index and nation.population > 0
        ]
        if not defensive_strengths:
            return None

        weakest_nation_index, weakest_nation_defensive_strength = min(
            defensive_strengths, key=lambda item: item[1]
        )

        if offensive_strength >= required_advantage * weakest_nation_defensive_strength:
            return weakest_nation_index

        return None

    def should_defend(self, nation_index: int) -> Optional[int]:
        """
        Maps caution, which is between 0 and 1, to the interval [0.5, 2]. That is the relative advantage
        the nation at the given nation_index aims to maintain in defense over its strongest neighbors
        offense.

        If it is not at this number, it shores up defensive resources and returns the largest threat's
        index. Otherwise, it returns None.
        """
        required_advantage = 0.5 + (1.5 * self.nations[nation_index].caution)

        defensive_strength = self.nations[nation_index].stocks[1]
        offensive_strengths = [
            (i, nation.stocks[0])
            for i, nation in enumerate(self.nations)
            if i != nation_index and nation.population > 0
        ]
        if not offensive_strengths:
            return None

        strongest_nation_index, strongest_nation_offensive_strength = max(
            offensive_strengths, key=lambda item: item[1]
        )

        if (
            defensive_strength
            <= required_advantage * strongest_nation_offensive_strength
        ):
            return strongest_nation_index

        return None

    def attack(self, attacker_index: int, defender_index: int) -> bool:
        """
        Resolves an attacker from the nation with attacker_index to the nation with defender_index.

        Requires an update of game state after the resolution of the attack.

        Returns True if attacker totally defeated the defender, and False otherwise.
        """
        attack_rating = self.nations[attacker_index].stocks[0]
        defense_rating = self.nations[defender_index].stocks[1]

        excess = attack_rating - defense_rating

        success = False

        if excess > 0:
            self.nations[attacker_index].stocks[0] = excess
            self.nations[defender_index].stocks[1] = 0

            initial_defender_procreative_stockpile = self.nations[
                defender_index
            ].stocks[3]
            if excess > initial_defender_procreative_stockpile:
                self.nations[defender_index].stocks[3] = 0
                self.nations[attacker_index].stocks[
                    0
                ] -= initial_defender_procreative_stockpile

                self.nations[defender_index].population -= self.nations[
                    attacker_index
                ].stocks[0]
                if self.nations[defender_index].population <= 0:
                    self.nations[defender_index].population = 0
                    success = True
                    for i in range(4):
                        self.nations[attacker_index].stocks[i] += self.nations[
                            defender_index
                        ].stocks[i]
                        self.nations[defender_index].stocks[i] = 0
            else:
                self.nations[defender_index].stocks[3] -= excess
                self.nations[attacker_index].stocks[0] = 0
        else:
            self.nations[defender_index].stocks[1] -= attack_rating
            self.nations[attacker_index].stocks[0] = 0
            # Note that excess is negative. We are reducing attacker's defensive resources first.
            # If defensive resources get depleted, this rolls over into population loss.
            self.nations[attacker_index].stocks[1] += excess
            if self.nations[attacker_index].stocks[1] < 0:
                self.nations[attacker_index].population += self.nations[
                    attacker_index
                ].stocks[1]
                self.nations[attacker_index].stocks[1] = 0
                if self.nations[attacker_index].population <= 0:
                    self.nations[attacker_index].population = 0
                    for i in range(4):
                        self.nations[defender_index].stocks[i] += self.nations[
                            attacker_index
                        ].stocks[i]
                        self.nations[attacker_index].stocks[i] = 0

        return success

    def trade(self, trader_index: int) -> bool:
        """
        This is how trading works:
        1. Order resources by priority for the nation at trader_index. If they are not in full defensive
        mode, the priority is determined by the ordering on their parameters. Otherwise defense is the only
        priority.

        2. See if there are any acceptable bids for that trade the current highest priorit resource for
        a resource of lower priority. If so, execute the trade for the maximum allowable number of units
        and return True. Otherwise, go to the next highest priority resource and repeat.

        3. If no trade is possible, return False.
        """
        trader = self.nations[trader_index]

        resources_by_priority = sorted(
            [
                (0, trader.aggression),
                (1, trader.caution),
                (2, trader.efficiency),
                (3, trader.fertility),
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        trader_parameters = [
            trader.aggression,
            trader.caution,
            trader.efficiency,
            trader.fertility,
        ]

        global_prices = np.array(
            [
                [
                    0.0
                    if (float(self.total_supply[supply]) == 0.0 or supply == demand)
                    else self.total_supply[demand] / self.total_supply[supply]
                    for supply in range(4)
                ]
                for demand in range(4)
            ]
        )
        beliefs = np.array(
            [
                [
                    0.0
                    if supply == demand
                    else trader_parameters[demand] / trader_parameters[supply]
                    for supply in range(4)
                ]
                for demand in range(4)
            ]
        )

        price_estimates = np.sqrt(np.multiply(global_prices, beliefs))

        # Elements of the form (asker_index, exchange_resource_index, exchange_amount, price, discount_from_estimate, resource_index)
        favorable_asks: List[Tuple[int, int, int, float, float]] = []

        for resource_index, _ in resources_by_priority:
            for exchange_resource_index, _ in resources_by_priority[
                resource_index + 1 :
            ]:
                if trader.stocks[exchange_resource_index] == 0:
                    continue

                for nation_index, nation in enumerate(self.nations):
                    if nation_index == trader_index:
                        continue

                    if nation.stocks[resource_index] == 0:
                        continue

                    nation_parameters = [
                        nation.aggression,
                        nation.caution,
                        nation.efficiency,
                        nation.fertility,
                    ]

                    if (
                        nation.asks[exchange_resource_index][resource_index]
                        >= price_estimates[exchange_resource_index][resource_index]
                    ):
                        nation_desired_proportion = (
                            nation_parameters[resource_index]
                            / nation_parameters[exchange_resource_index]
                        )

                        # Solve for n:
                        # (nation.stocks[resource_index] - n) / (nation.stocks[exchange_resource_index] + n*nation.asks[resource_index][exchange_resource_index]) = nation_desired_proportion
                        max_amount = int(
                            (
                                nation.stocks[resource_index]
                                - nation_desired_proportion
                                * nation.stocks[exchange_resource_index]
                            )
                            / (
                                1
                                + (
                                    nation.asks[resource_index][exchange_resource_index]
                                    * nation_desired_proportion
                                )
                            )
                        )
                        if max_amount < 0:
                            # This means that the nation does not have a surplus of the resource at resource_index.
                            continue
                        max_amount = min(max_amount, nation.stocks[resource_index])

                        favorable_asks.append(
                            (
                                nation_index,
                                exchange_resource_index,
                                max_amount,
                                nation.asks[exchange_resource_index][resource_index],
                                nation.asks[exchange_resource_index][resource_index]
                                / price_estimates[exchange_resource_index][
                                    resource_index
                                ],
                                resource_index,
                            )
                        )

        if not favorable_asks:
            return False

        # Elements of the form (asker_index, exchange_resource_index, exchange_amount, price, discount_from_estimate, resource_index)
        # Favorability of trade depends on how much trader favors hoarding or trade.
        # Sort available bids of desired resource by favorability.
        favorable_asks.sort(
            key=lambda item: trader.hoarder
            * (trader.stocks[item[1]] + item[2])
            / (sum(trader.stocks) + 1)
            + trader.trader * item[4],
            reverse=True,
        )

        (
            asker_index,
            trade_exchange_resource_index,
            max_trade_amount,
            price,
            _,
            trade_resource_index,
        ) = favorable_asks[0]

        asker = self.nations[asker_index]

        max_resource_tradable = min(
            max_trade_amount, trader.stocks[trade_resource_index]
        )
        if max_resource_tradable * price > asker.stocks[trade_exchange_resource_index]:
            max_resource_tradable = int(
                asker.stocks[trade_exchange_resource_index] / price
            )

        trader.stocks[trade_resource_index] -= max_resource_tradable
        asker.stocks[trade_resource_index] += max_resource_tradable

        trader.stocks[trade_exchange_resource_index] += int(
            max_resource_tradable * price
        )
        asker.stocks[trade_exchange_resource_index] -= int(
            max_resource_tradable * price
        )

        if (
            trader.stocks[trade_resource_index] < 0
            or trader.stocks[trade_exchange_resource_index] < 0
            or asker.stocks[trade_resource_index] < 0
            or asker.stocks[trade_exchange_resource_index] < 0
        ):
            breakpoint()

        return True

    def harvest(self, nation_index: int, weights: Optional[List[float]] = None) -> bool:
        """
        If weights are provided they will be used to determine harvesting allocation. Otherwise, it
        will be done based on nation's parameters.
        """
        nation = self.nations[nation_index]
        if nation.population == 0:
            return False

        if weights is None:
            weights = [
                nation.aggression,
                nation.caution,
                nation.efficiency,
                nation.fertility,
            ]

        # Some leeway for rounding effects on sum.
        assert sum(weights) <= 1.01, "Weights are too high"
        for weight in weights:
            assert weight >= 0, "Weights must be non-negative"

        non_procreative_harvesting = [
            int(nation.population * weights[0]),
            int(nation.population * weights[1]),
            int(nation.population * weights[2]),
        ]
        procreative_harvesting = nation.population - sum(non_procreative_harvesting)
        harvesting = non_procreative_harvesting + [procreative_harvesting]

        amount_harvested = [
            int(harvesting_population * nation.efficiency)
            for harvesting_population in harvesting
        ]
        for i, amount in enumerate(amount_harvested):
            nation.stocks[i] += amount

        return True

    def increase_efficiency(self, nation_index: int) -> bool:
        nation = self.nations[nation_index]
        if nation.stocks[2] < nation.technology + 1:
            return False

        nation.stocks[2] -= nation.technology + 1
        nation.technology += 1
        return True

    def increase_populations(self):
        for nation in self.nations:
            population_increase = min(
                int(nation.fertility * nation.population), nation.stocks[3]
            )
            nation.stocks[3] -= population_increase
            nation.population += population_increase

    def print_status(self, show_asks: bool = False) -> None:
        print(f"Tick: {self.last_tick}")
        for i, nation in enumerate(self.nations):
            print(f"Nation {i}:")
            print(
                f"Mentality: builder={nation.builder}, hoarder={nation.hoarder}, trader={nation.trader}"
            )
            print(
                f"Parameters: aggression={nation.aggression}, caution={nation.caution}, efficiency={nation.efficiency}, fertility={nation.fertility}"
            )
            print(f"\tPopulation: {nation.population}")
            print(f"\tTechnology: {nation.technology}")
            print(f"\tStocks: {json.dumps(nation.stocks)}")
            if show_asks:
                print("\tAsks:")
                for i in range(4):
                    print(f"\t\tResource {i}: {nation.asks[i]}")

    def tick(self, shuffle_nations: bool = True, output: bool = True) -> None:
        """
        Simulates one tick of the game.
        """
        eliminated = 0
        self.last_tick += 1

        if output:
            self.print_status()

        if self.last_tick % self.population_growth_interval == 0:
            self.increase_populations()

        national_indices = list(range(len(self.nations)))
        if shuffle_nations:
            random.shuffle(national_indices)

        for nation_index in national_indices:
            if self.nations[nation_index].population == 0:
                eliminated += 1
                continue

            target = self.should_attack(nation_index)
            if target is not None:
                result = self.attack(nation_index, target)
                print(f"ATTACK: {nation_index} -> {target}")
                if result:
                    print(f"ELIMINATED: {target}")
                if self.nations[nation_index].population == 0:
                    print(f"ELIMINATED: {nation_index}")
                continue

            # 0 for improve_efficiency, 1 for harvest, 2 for trade
            choice = random.random()
            mentality_cdf = [
                self.nations[nation_index].builder,
                self.nations[nation_index].builder + self.nations[nation_index].hoarder,
                1,
            ]

            if choice <= mentality_cdf[0]:
                economic_action = 0
            elif choice <= mentality_cdf[1]:
                economic_action = 1
            else:
                economic_action = 2

            harvesting_weights: Optional[List[float]] = None

            # But only if we shouldn't be defending
            threat = self.should_defend(nation_index)
            if threat is not None:
                harvesting_weights = [0.0, 0.5, 0.0, 0.5]

                # Zero out all bids that use up defensive resources
                for i in range(4):
                    self.nations[nation_index].asks[1][i] = 0.0
                economic_action = 2

            if economic_action == 0:
                result = self.increase_efficiency(nation_index)
                if not result:
                    economic_action = 2

            if economic_action == 1:
                result = self.trade(nation_index)
                if not result:
                    economic_action = 2

            if economic_action == 2:
                self.harvest(nation_index, harvesting_weights)

        if output:
            self.print_status()

        if eliminated == len(self.nations):
            raise Exception("All nations have been eliminated")

        self.update()

    def rollout(
        self, ticks: int, shuffle_nations: bool = True, output: bool = True
    ) -> None:
        """
        Simulates a number of ticks of the game.
        """
        for _ in range(ticks):
            self.tick(shuffle_nations=shuffle_nations, output=output)
