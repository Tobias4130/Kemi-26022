# equilibrium.py
import re
from typing import Dict
from compound import Compound
from equations import ChemicalEquation

def calculate_kp(
    reaction: ChemicalEquation,
    partial_pressures: Dict[Compound, float]
) -> float:
    """
    Calculates the equilibrium constant Kp from equilibrium partial pressures.
    """
    # Get the balanced coefficients for the reaction
    reactant_coeffs, product_coeffs = reaction.balance()
    
    # Calculate the product term
    products_term = 1.0
    for compound, coeff in zip(reaction.products, product_coeffs):
        products_term *= partial_pressures[compound] ** coeff
        
    # Calculate the reactant term
    reactants_term = 1.0
    for compound, coeff in zip(reaction.reactants, reactant_coeffs):
        reactants_term *= partial_pressures[compound] ** coeff
        
    if reactants_term == 0:
        raise ValueError("The product of reactant pressures cannot be zero.")
        
    return products_term / reactants_term

def calculate_kp_generic(
    reactants: dict[str, int],
    products: dict[str, int],
    partial_pressures: dict[str, float]
) -> float:
    """
    Calculates Kp from dictionaries of species, coefficients, and partial pressures.
    
    Args:
        reactants: A dictionary of {species_name: coefficient}.
        products: A dictionary of {species_name: coefficient}.
        partial_pressures: A dictionary of {species_name: pressure}.
    """
    products_term = 1.0
    for species, coeff in products.items():
        products_term *= partial_pressures[species] ** coeff
        
    reactants_term = 1.0
    for species, coeff in reactants.items():
        reactants_term *= partial_pressures[species] ** coeff
        
    if reactants_term == 0:
        raise ValueError("The product of reactant pressures cannot be zero.")
        
    return products_term / reactants_term

def manipulate_equilibrium_constant(
    original_k: float,
    is_reversed: bool,
    multiplier: float
) -> float:
    """
    Calculates a new equilibrium constant after manipulating a reaction.
    """
    new_k = original_k
    
    if is_reversed:
        new_k = 1 / new_k
    
    if multiplier != 1:
        new_k = new_k ** multiplier
        
    return new_k

def calculate_kc(
    reaction: ChemicalEquation,
    concentrations: Dict[Compound, float]
) -> float:
    """
    Calculates the equilibrium constant Kc from equilibrium concentrations.
    Automatically excludes pure solids (s) and liquids (l).
    """
    reactant_coeffs, product_coeffs = reaction.balance()
    
    products_term = 1.0
    for compound, coeff in zip(reaction.products, product_coeffs):
        if compound.phase not in ['s', 'l']:
            products_term *= concentrations[compound] ** coeff
            
    reactants_term = 1.0
    for compound, coeff in zip(reaction.reactants, reactant_coeffs):
        if compound.phase not in ['s', 'l']:
            reactants_term *= concentrations[compound] ** coeff
            
    if reactants_term == 0:
        raise ValueError("The product of reactant concentrations cannot be zero.")
        
    return products_term / reactants_term

def solve_for_equilibrium_concentration(
    reaction: ChemicalEquation,
    k_value: float,
    known_concentrations: Dict[Compound, float]
) -> float:
    """
    Solves for a single unknown equilibrium concentration given Kc or Kp.
    """
    reactant_coeffs, product_coeffs = reaction.balance()
    
    # Calculate the product and reactant terms from known values
    products_term = 1.0
    unknown_species = None
    unknown_coeff = 1
    
    for compound, coeff in zip(reaction.products, product_coeffs):
        if compound.phase not in ['s', 'l']:
            if compound in known_concentrations:
                products_term *= known_concentrations[compound] ** coeff
            else:
                unknown_species = ('product', compound)
                unknown_coeff = coeff

    reactants_term = 1.0
    for compound, coeff in zip(reaction.reactants, reactant_coeffs):
        if compound.phase not in ['s', 'l']:
            if compound in known_concentrations:
                reactants_term *= known_concentrations[compound] ** coeff
            else:
                unknown_species = ('reactant', compound)
                unknown_coeff = coeff
                
    # Solve for the unknown
    if unknown_species[0] == 'reactant':
        # K = products / (reactants * unknown^coeff) -> unknown^coeff = products / (K * reactants)
        result = (products_term / (k_value * reactants_term))**(1/unknown_coeff)
    else: # unknown is a product
        # K = (products * unknown^coeff) / reactants -> unknown^coeff = (K * reactants) / products
        result = ((k_value * reactants_term) / products_term)**(1/unknown_coeff)
        
    return result

def calculate_molar_solubility(formula: str, ksp: float) -> tuple[float, dict[str, float]]:
    """
    Calculates molar solubility and ion concentrations from Ksp.
    Assumes dissociation of a simple salt AxBy.
    """
    # This is a simplified parser for formulas like 'CaF2' or 'Ag2SO4'
    parts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    
    if len(parts) == 2: # Simple binary salt AxBy
        cation_sym, x_str = parts[0]
        anion_sym, y_str = parts[1]
        x = int(x_str) if x_str else 1
        y = int(y_str) if y_str else 1
    else: # Fallback for more complex cases, may need refinement
        # This is a simplified assumption for this function
        return (ksp, {})

    # Ksp = (x*s)^x * (y*s)^y = (x^x * y^y) * s^(x+y)
    s = (ksp / ((x**x) * (y**y))) ** (1 / (x + y))
    
    ion_concentrations = {
        cation_sym: x * s,
        anion_sym: y * s
    }
    
    return s, ion_concentrations

R_GAS_CONSTANT_ATM = 0.08206 # L·atm/(mol·K)

def convert_kp_to_kc(
    kp_value: float,
    temp_K: float,
    delta_n: int
) -> float:
    """
    Converts an equilibrium constant Kp to Kc using the provided Δn.
    """
    # Kc = Kp / (RT)^Δn
    kc = kp_value / ((R_GAS_CONSTANT_ATM * temp_K)**delta_n)
    return kc

R_GAS_CONSTANT_BAR = 0.083145 # L·bar/(mol·K)

def convert_kc_to_kp(
    kc_value: float,
    temp_K: float,
    delta_n: int
) -> float:
    """
    Converts an equilibrium constant Kc to Kp using the provided Δn.
    
    Args:
        kc_value: The value of Kc.
        temp_K: The temperature in Kelvin.
        delta_n: The change in moles of gas (products - reactants).
    """
    # Kp = Kc(RT)^Δn
    kp = kc_value * ((R_GAS_CONSTANT_BAR * temp_K)**delta_n)
    return kp

def solve_for_equilibrium_pressure(
    reaction: ChemicalEquation,
    kp_value: float,
    known_pressures: Dict[Compound, float]
) -> float:
    """
    Solves for a single unknown equilibrium partial pressure given Kp.
    """
    reactant_coeffs, product_coeffs = reaction.balance()
    
    products_term = 1.0
    unknown_species = None
    unknown_coeff = 1
    
    for compound, coeff in zip(reaction.products, product_coeffs):
        if compound.phase not in ['s', 'l']:
            if compound in known_pressures:
                products_term *= known_pressures[compound] ** coeff
            else:
                unknown_species = ('product', compound)
                unknown_coeff = coeff

    reactants_term = 1.0
    for compound, coeff in zip(reaction.reactants, reactant_coeffs):
        if compound.phase not in ['s', 'l']:
            if compound in known_pressures:
                reactants_term *= known_pressures[compound] ** coeff
            else:
                unknown_species = ('reactant', compound)
                unknown_coeff = coeff
                
    if unknown_species[0] == 'reactant':
        result = (products_term / (kp_value * reactants_term))**(1/unknown_coeff)
    else:
        result = ((kp_value * reactants_term) / products_term)**(1/unknown_coeff)
        
    return result

def calculate_ksp_from_solubility(
    formula: str,
    solubility_g_per_L: float
) -> float:
    """
    Calculates the Ksp value from the solubility of a salt in g/L.
    Handles simple and common polyatomic ionic compounds.
    """
    # 1. Calculate molar solubility (s)
    solute = Compound(formula)
    molar_solubility_s = solubility_g_per_L / solute.molar_mass
    
    # 2. Parse the formula to get coefficients x and y
    # This logic is adapted from your more robust solutions.py parser
    composition = solute.composition.copy()
    ion_coeffs = []
    
    # Check for common polyatomic ions first
    polyatomic_ions = ['NH4', 'OH', 'NO3', 'SO4', 'CO3', 'PO4']
    found_poly = False
    for ion in polyatomic_ions:
        poly_comp = Compound(ion).composition
        if all(composition.get(atom, 0) >= num for atom, num in poly_comp.items()):
            # Find how many times this ion can be formed
            num_ions = min(composition[atom] // num for atom, num in poly_comp.items())
            ion_coeffs.append(num_ions)
            # Subtract the atoms of the polyatomic ion
            for atom, num in poly_comp.items():
                composition[atom] -= num * num_ions
                if composition[atom] == 0:
                    del composition[atom]
            found_poly = True
            break # Assume only one type of polyatomic ion for simplicity in Ksp

    # The remaining atoms are the counter-ion
    for atom, count in composition.items():
        ion_coeffs.append(count)
        
    if len(ion_coeffs) != 2:
        raise ValueError(f"Could not parse '{formula}' into two distinct ions.")
    
    x, y = ion_coeffs[0], ion_coeffs[1]

    # 3. Calculate Ksp using the formula: Ksp = (x^x * y^y) * s^(x+y)
    ksp = (x**x) * (y**y) * (molar_solubility_s**(x + y))
    
    return ksp

