from crime_location import CrimeLocationAnalysis
from crime_type_and_weapon import CrimeTypeAndWeapon
from crime_recidivism import CrimeRecidivism


def main():
    # analysis = CrimeTypeAndWeapon()
    analysis = CrimeLocationAnalysis()
    # analysis = CrimeRecidivism('updated_dataset_with_uneven_crime_types.csv')
    
    analysis.execute()


if __name__ == "__main__":
    main()
