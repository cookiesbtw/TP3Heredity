import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    p = 1.0

    print("people", people, flush=True)
    print("one_gene", one_gene, flush=True)
    print("two_genes", two_genes, flush=True)
    print("have_trait", have_trait, flush=True)


    for person in people:
        mother = people[person]["mother"]
        father = people[person]["father"]

        # Determine number of genes
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Determine trait
        has_trait = person in have_trait

        # Calculate probability
        if mother is None and father is None:
            # No parents information
            gene_prob = PROBS["gene"][genes]
        else:
            # With parents information
            # Calculate probability of inheriting gene from each parent
            def parent_pass_gene_prob(parent):
                if parent in two_genes:
                    print("parent", parent, "passes gene with prob", 1 - PROBS["mutation"], flush=True)
                    return 1 - PROBS["mutation"]        
                elif parent in one_gene:
                    return 0.5
                else:
                    return PROBS["mutation"]
                
            mother_prob = parent_pass_gene_prob(mother)
            father_prob = parent_pass_gene_prob(father)

            if genes == 2:
                gene_prob = mother_prob * father_prob
            elif genes == 1:
                gene_prob = mother_prob * (1 - father_prob) + (1 - mother_prob) * father_prob
            else:
                gene_prob = (1 - mother_prob) * (1 - father_prob)

        trait_prob = PROBS["trait"][genes][has_trait]

        p *= gene_prob * trait_prob

    return p

    # raise NotImplementedError


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    for person in probabilities:
        # Determine number of genes
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Determine trait
        has_trait = person in have_trait

        # Update probabilities
        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][has_trait] += p

    print("Updated probabilities:", probabilities, flush=True)
    

    # raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


# AI generated test code
if __name__ == "__main__":
    PROBS = {
        "gene": {2: 0.01, 1: 0.03, 0: 0.96},
        "trait": {
            2: {True: 0.65, False: 0.35},
            1: {True: 0.56, False: 0.44},
            0: {True: 0.01, False: 0.99}
        },
        "mutation": 0.01
    }

    people = {
        "Harry": {"name": "Harry", "mother": None, "father": None, "trait": None}
    }

    one_gene = {"Harry"}
    two_genes = set()
    have_trait = {"Harry"}

    # Dicionário inicial de probabilidades
    probabilities = {
        "Harry": {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0}
        }
    }

    # 1️⃣ Calcular probabilidade conjunta
    p = joint_probability(people, one_gene, two_genes, have_trait)
    print("\nJoint probability p =", p)

    # 2️⃣ Mostrar antes do update
    print("\nBefore update:")
    for person in probabilities:
        print(f"{person}: {probabilities[person]}")

    # 3️⃣ Atualizar distribuição
    update(probabilities, one_gene, two_genes, have_trait, p)

    # 4️⃣ Mostrar depois do update
    print("\nAfter update:")
    for person in probabilities:
        print(f"{person}: {probabilities[person]}")


