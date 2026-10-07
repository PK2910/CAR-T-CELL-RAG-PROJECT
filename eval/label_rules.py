"""
Answer rules used to label which chunks ANSWER each exam question.

A chunk counts as relevant if, for at least one rule of the question:
  - some single SENTENCE in it matches every regex in "sentence", and
  - the chunk as a whole matches every regex in "context" (optional).
All regexes are case-insensitive. Rules were written by reading samples of
their matches (see auto_label.py) and checking they really contain answers.
"""

PROD_TISA = r"tisagenlecleucel|tisa-cel|kymriah|ctl019|eliana"
PROD_BREXU = r"zuma-3|brexucabtagene|brexu-cel|kte-x19|tecartus"
PROD_OBE = r"obecabtagene|obetacabtagene|obe-cel|auto1\b|felix|aucatzyl"
ALL_CONTEXT = r"lymphoblastic|b-all|\ball\b"
PCT = r"\d\s?%"
SEVERE = r"severe|grade\s*(≥|>=)\s*3|≥\s*grade\s*3|grade\s*3|grade\s*iii|grade\s*4|grades?\s*3\s*(-|–|to|/|and|or)\s*4"

RULES = {
    "q01": [{"sentence": [r"eliana|tisagenlecleucel|tisa-cel|ctl019", r"\b8[12](\.\d)?\s?%",
                          r"remission|response|\bCR\b|ORR"]}],
    "q02": [{"sentence": [PROD_BREXU, r"\b(71|69|56|53)(\.\d)?\s?%"], "context": [r"remission|response|\bCR\b|ORR"]}],
    # The product name must be close to the percentage (long run-on sentences mix trials)
    "q03": [{"sentence": [r"(" + PROD_OBE + r"|cohort iia).{0,150}\b7[67](\.6)?\s?%|\b7[67](\.6)?\s?%.{0,150}(" + PROD_OBE + ")"]}],
    "q04": [{"sentence": [r"severe CRS|CRS.{0,40}grade\s*(≥|>=|3)|grade\s*(≥|>=)?\s*3.{0,40}CRS|≥\s*grade\s*3\s*CRS",
                          r"incidence|occurred|developed|experienced|rate|range|in \d+(\.\d)?\s?%", PCT],
             "context": [ALL_CONTEXT]}],
    "q05": [{"sentence": [r"ICANS|neurotoxicit|neurologic(al)?\s+(events|toxicit)",
                          r"incidence|occurred|occurs|developed|experienced|reported|rate|range", PCT],
             "context": [ALL_CONTEXT]}],
    "q06": [{"sentence": [r"real.world|real.life|cibmtr|prwcc|registry"], "context": [PROD_TISA, r"real.world|real.life|cibmtr|prwcc"]}],
    "q07": [{"sentence": [r"CD22", r"\bCR\b|remission|response", PCT], "context": [r"CAR"]}],
    "q08": [{"sentence": [r"event.free|\bEFS\b|relapse.free|\bRFS\b", PCT, r"month|year"],
             "context": [PROD_TISA + r"|pediatric|paediatric|children"]}],
    "q09": [{"sentence": [r"CD19.?(negative|neg\b|-\s|loss)|antigen\s+(loss|escape)|CD19\s+escape",
                          r"splic|exon|lineage switch|mutation|trogocytosis|myeloid|epitope|frameshift"]}],
    "q10": [{"sentence": [r"CD19.?(positive|\+)", r"relapse", r"persist|exhaust|b.cell recovery|loss of CAR|functional|durab"]}],
    "q11": [{"sentence": [r"4-1BB|41BB|4‐1BB", r"CD28", r"persist|expan|exhaust|memory|kinetic|durab"]}],
    "q12": [{"sentence": [r"lymphodeplet", r"fludarabine|cyclophosphamide|flu/cy|flu-cy"]},
            {"sentence": [r"lymphodeplet", r"homeostatic|IL-7|IL-15|regulatory T|expansion|engraft|persistence"]}],
    "q13": [{"sentence": [r"b.cell aplasia|hypogammaglobulin", r"immunoglobulin|IVIG|infection|persist|function|surrogate|marker"]}],
    # "(?<!young )adult" so the phrase "children and young adults" doesn't count as a comparison
    "q14": [{"sentence": [r"(?<!young )adults?\b", r"pediatric|paediatric|children|younger patients",
                          r"CRS|toxicit|remission|survival|outcome|relapse|\bEFS\b|\bCR\b",
                          r"compar|whereas|than|versus|\bvs\b|differ|higher|lower|similar"]}],
    "q15": [{"sentence": [r"transplant|HSCT|allo-?HCT|allo-?SCT|\bHCT\b|\bSCT\b",
                          r"consolidat|after CAR|post-CAR|following CAR|bridge to|subsequent"],
             "context": [r"CAR"]}],
    "q16": [{"sentence": [r"blinatumomab|inotuzumab", r"CAR", r"compar|versus|\bvs\.?|than|whereas|prior (exposure|treatment|blinatumomab|inotuzumab)"]}],
    "q17": [{"sentence": [r"fratricide|t.cell aplasia|contaminat|shared antigen|CD7|CD5"],
             "context": [r"T-ALL|T.cell acute lymphoblastic"]}],
    "q18": [{"sentence": [r"ICANS|neurotoxicit", r"\bICE\b|ASTCT|grading|graded"]},
            {"sentence": [r"ICANS|neurotoxicit", r"dexamethasone|corticosteroid|steroid|methylprednisolone|anakinra"]}],
    "q19": [{"sentence": [r"\bHLH\b|hemophagocytic|haemophagocytic|\bMAS\b|macrophage activation|IEC-HS",
                          r"ferritin|criteria|defined|characteri|anakinra|etoposide|ruxolitinib"]}],
    "q20": [{"sentence": [r"cytopeni", r"prolonged|late|biphasic|hematotox|G-CSF|CD34|boost|thrombopoietin|eltrombopag|inflammat"]}],
    "q21": [{"sentence": [r"approv", r"\b25\b|≤\s?25|26 years|younger than 26"], "context": [PROD_TISA]}],
    # Dose research questions (answerable; doses are fine here because no individual patient is involved)
    "q29": [{"sentence": [r"tocilizumab", r"\d+\s?mg\s?/\s?kg"]}],
    # ZUMA-2 is brexu-cel's mantle cell lymphoma trial (a different disease and dose), so it is excluded
    "q30": [{"sentence": [r"\d(\.\d)?\s?[x×]\s?10\s?\^?\s?6|10\s?\^?\s?6\s?(CAR|cells)", r"cells|CAR", r"^(?!.*ZUMA-2)"],
             "context": [PROD_BREXU, ALL_CONTEXT]}],
    # Expected to have no answer in the corpus (checked, not assumed):
    "q22": [{"sentence": [r"2027", r"phase\s*(3|iii)|first.line|frontline|newly diagnosed"]}],
    "q23": [{"sentence": [r"\bdiet|nutrition|malnutrition|\bfeeding"]}],
    "q24": [{"sentence": [r"canberra|capital of australia"]}],
    "q25": [{"sentence": [r"tocilizumab", r"mg/kg|\bdose\b|dosing"]}],
}
