import csv
import itertools
import sys

# PROBS: model constants — prior probabilities and mutation/trait rates
PROBS = {

    # Unconditional probabilities of having 0, 1 or 2 gene copies
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    # Probability of displaying the trait given number of genes
    "trait": {

        # Probability of trait with 2 copies
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait with 1 copy
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait with 0 copies
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability when a parent passes the gene to a child
    "mutation": 0.01
}


def main():
    # Main function: read data, compute joint probabilities and normalize

    # Check correct usage (one CSV filename argument)
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Structure to accumulate distributions for each person
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

    # Names as a set to generate subsets (powerset)
    names = set(people)
    for have_trait in powerset(names):

        # Skip combinations that conflict with known evidence (trait column)
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Test all combinations of who has 1 gene and who has 2 genes
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Compute joint probability for this configuration
                p = joint_probability(people, one_gene, two_genes, have_trait)
                # Update accumulated distributions with p
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Normalize so each distribution sums to 1
    normalize(probabilities)

    # Print final results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Read CSV and return dict:
    { name: {"name": name, "mother": mother or None, "father": father or None, "trait": True/False/None} }
    Expected CSV format: name,mother,father,trait
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                # empty string -> None (no parent information)
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                # trait: "1" -> True, "0" -> False, empty -> None
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all subsets of set s.
    Useful to enumerate all possible combinations of who has the gene/trait.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute the joint probability of the given configuration:
      - people in one_gene have 1 copy of the gene
      - people in two_genes have 2 copies
      - the rest have 0 copies
      - people in have_trait display the trait, the rest do not
    Processes each person independently multiplying individual probabilities,
    taking into account inheritance from parents when available and mutation rate.
    """

    p = 1.0

    # Debug prints showing the current configuration
    print("people", people, flush=True)
    print("one_gene", one_gene, flush=True)
    print("two_genes", two_genes, flush=True)
    print("have_trait", have_trait, flush=True)

    for person in people:
        mother = people[person]["mother"]
        father = people[person]["father"]

        # Determine how many gene copies 'person' has in this configuration
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Whether the person has the trait in this configuration
        has_trait = person in have_trait

        # Compute probability of having that number of genes
        if mother is None and father is None:
            # No parent information: use unconditional probability
            gene_prob = PROBS["gene"][genes]
        else:
            # With parents: compute probability each parent passes the gene
            def parent_pass_gene_prob(parent):
                # If parent has 2 genes: passes gene with prob 1 - mutation
                if parent in two_genes:
                    # debug print
                    print("parent", parent, "passes gene with prob", 1 - PROBS["mutation"], flush=True)
                    return 1 - PROBS["mutation"]
                # If parent has 1 gene: passes with prob 0.5
                elif parent in one_gene:
                    return 0.5
                # If parent has 0 genes: only passes if mutation occurs
                else:
                    return PROBS["mutation"]
                
            mother_prob = parent_pass_gene_prob(mother)
            father_prob = parent_pass_gene_prob(father)

            # Probability child has 0/1/2 copies given each parent's pass probability
            if genes == 2:
                gene_prob = mother_prob * father_prob
            elif genes == 1:
                gene_prob = mother_prob * (1 - father_prob) + (1 - mother_prob) * father_prob
            else:
                gene_prob = (1 - mother_prob) * (1 - father_prob)

        # Probability of displaying (or not) the trait given gene count
        trait_prob = PROBS["trait"][genes][has_trait]

        # Multiply into the joint product
        p *= gene_prob * trait_prob

    return p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add joint probability p to the marginal distributions in `probabilities`.
    For each person, add p to the correct category (0/1/2 genes and True/False trait).
    """

    for person in probabilities:
        # Determine number of genes for this person in the current configuration
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Determine whether they have the trait in this configuration
        has_trait = person in have_trait

        # Update accumulators
        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][has_trait] += p

    # Print to help observe updates during execution
    print("Updated probabilities:", probabilities, flush=True)
    

def normalize(probabilities):
    """
    Normalize the distributions in `probabilities` so each distribution
    (gene and trait for each person) sums exactly to 1, preserving proportions.
    """

    for person in probabilities:
        # Normalize gene: divide each entry by total sum
        gene_total = sum(probabilities[person]["gene"].values())
        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= gene_total

        # Normalize trait
        trait_total = sum(probabilities[person]["trait"].values())
        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] /= trait_total


# AI Generated
# Test code (runs when this file is executed directly).
# Keeps a simple example of one person with no known parents.
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

    probabilities = {
        "Harry": {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0}
        }
    }

    # 1️⃣ Compute joint probability
    p = joint_probability(people, one_gene, two_genes, have_trait)
    print("\nJoint probability p =", p)

    # 2️⃣ Show before update
    print("\nBefore update:")
    for person in probabilities:
        print(f"{person}: {probabilities[person]}")

    # 3️⃣ Update distribution
    update(probabilities, one_gene, two_genes, have_trait, p)

    # 4️⃣ Show after update
    print("\nAfter update:")
    for person in probabilities:
        print(f"{person}: {probabilities[person]}")

    # 5️⃣ Normalize
    normalize(probabilities)

    # 6️⃣ Show after normalization
    print("\nAfter normalization:")
    for person in probabilities:
        print(f"{person}: {probabilities[person]}")



