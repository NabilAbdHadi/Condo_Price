from numpy.lib.function_base import diff
import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CondoPrice():

    def __init__(self, demandLocation, numberOfBedroom,
                    numberOfBathroom, houseFunishing, 
                    houseAreaSize, providedFacilities, 
                    accessArea):
        
        if(houseFunishing == 'unfunished'):
            houseFunishing = 0
        elif(houseFunishing == 'partially'):
            houseFunishing = 1
        elif(houseFunishing == 'fully'):
            houseFunishing = 2
        
        self.demandLocation = demandLocation
        self.numberOfBedroom = numberOfBedroom
        self.numberOfBathroom = numberOfBathroom
        self.houseFunishing = houseFunishing     
        self.houseAreaSize = houseAreaSize
        self.providedFacilities = providedFacilities
        self.accessArea = accessArea

        """ declare the range for each input """
        self.location = np.arange(0,4000,400)
        self.bedroom = np.arange(0,11,1.0)
        self.bathroom = np.arange(0,11,1.0)
        self.facilities = np.arange(0,11,1.0)
        self.funishing = np.arange(0,3,1.0)
        self.areaSize = np.arange(0,3000,200)
        self.accessibility = np.arange(0,11,1.0)
        self.psf = np.arange(0,1000,100)

    def membershipFunction(self):
        """ initialize all membership functions """
        """ Location MF """
        self.very_less_demand = fuzz.trapmf(self.location,[0,0,500,600])
        self.less_demand = fuzz.trapmf(self.location,[500,600,1000,1100])
        self.average_demand = fuzz.trapmf(self.location,[1000,1100,1500,1600])
        self.high_demand = fuzz.trapmf(self.location,[1500,1600,2000,2100])
        self.very_high_demand = fuzz.trapmf(self.location,[2000,2100,4000,4000])

        """ Bedroom MF """
        self.less_bed = fuzz.trapmf(self.bedroom,[0,0,2,3])
        self.average_bed = fuzz.trimf(self.bedroom,[2,3,4])
        self.more_bed = fuzz.trapmf(self.bedroom,[3,4,11,11])

        """ Bathroom MF """
        self.less_bath = fuzz.trapmf(self.bathroom,[0,0,1,2])
        self.average_bath = fuzz.trimf(self.bathroom,[1,2,3])
        self.more_bath = fuzz.trapmf(self.bathroom,[2,3,11,11])

        """ Facilities MF """
        self.less_fac = fuzz.trapmf(self.facilities,[0,0,3,4])
        self.average_fac = fuzz.trapmf(self.facilities,[3,4,6,7])
        self.high_fac = fuzz.trapmf(self.facilities,[6,7,11,11])

        """ Funishing MF """
        self.unfun = fuzz.trimf(self.funishing,[0,0,1])
        self.partially_fun = fuzz.trimf(self.funishing,[0,1,2])
        self.full_fun = fuzz.trimf(self.funishing,[1,2,3])

        """ Area size MF """
        self.very_small_area = fuzz.trapmf(self.areaSize,[0,0,500, 600])
        self.small_area = fuzz.trapmf(self.areaSize,[500, 600, 900, 1000])
        self.average_area = fuzz.trapmf(self.areaSize,[900, 1000, 1200, 1300])
        self.large_area = fuzz.trapmf(self.areaSize,[1200, 1300, 3100, 3100]) 

        """ Accessibility MF """
        self.bad_access = fuzz.trapmf(self.accessibility,[0,0,4,6])
        self.average_access = fuzz.trimf(self.accessibility, [4,5,6])
        self.good_access = fuzz.trapmf(self.accessibility,[4,6,11,11])
        
        """ PSF MF"""
        self.below_price = fuzz.trapmf(self.psf, [0,100,200,300])
        self.standard_price = fuzz.trapmf(self.psf, [200,300,400,500])
        self.above_price = fuzz.trapmf(self.psf, [400, 500, 700, 800])
        self.high_price = fuzz.trapmf(self.psf, [700,  800, 900, 1000])

    def interpretingMF(self):
        """ interpreting the membership function between the input and membership function """
        self.location_is_veryLessDemand = fuzz.interp_membership(self.location, self.very_less_demand, self.demandLocation)
        self.location_is_lessDemand = fuzz.interp_membership(self.location, self.less_demand, self.demandLocation)
        self.location_is_averageDemand = fuzz.interp_membership(self.location, self.average_demand, self.demandLocation)
        self.location_is_highDemand = fuzz.interp_membership(self.location, self.high_demand, self.demandLocation)
        self.location_is_veryHighDemand = fuzz.interp_membership(self.location, self.very_high_demand, self.demandLocation)

        self.bed_is_less = fuzz.interp_membership(self.bedroom, self.less_bed, self.numberOfBedroom)
        self.bed_is_average = fuzz.interp_membership(self.bedroom, self.average_bed, self.numberOfBedroom)
        self.bed_is_more = fuzz.interp_membership(self.bedroom, self.more_bed, self.numberOfBedroom)

        self.bath_is_less = fuzz.interp_membership(self.bathroom, self.less_bath, self.numberOfBathroom)
        self.bath_is_average = fuzz.interp_membership(self.bathroom, self.average_bath, self.numberOfBathroom)
        self.bath_is_more = fuzz.interp_membership(self.bathroom, self.more_bath, self.numberOfBathroom)

        self.fac_is_low = fuzz.interp_membership(self.facilities, self.less_fac, self.providedFacilities)
        self.fac_is_average = fuzz.interp_membership(self.facilities, self.average_fac, self.providedFacilities)
        self.fac_is_high = fuzz.interp_membership(self.facilities, self.high_fac, self.providedFacilities)

        self.unfunishing = fuzz.interp_membership(self.funishing, self.unfun, self.houseFunishing)
        self.partially_funishing = fuzz.interp_membership(self.funishing, self.partially_fun, self.houseFunishing)
        self.fully_funishing = fuzz.interp_membership(self.funishing, self.full_fun, self.houseFunishing)

        self.area_is_verysmall = fuzz.interp_membership(self.areaSize, self.very_small_area, self.houseAreaSize)
        self.area_is_small = fuzz.interp_membership(self.areaSize, self.small_area, self.houseAreaSize)
        self.area_is_average = fuzz.interp_membership(self.areaSize, self.average_area, self.houseAreaSize)
        self.area_is_large = fuzz.interp_membership(self.areaSize, self.large_area, self.houseAreaSize)

        self.access_is_bad = fuzz.interp_membership(self.accessibility, self.bad_access, self.accessArea)
        self.access_is_average = fuzz.interp_membership(self.accessibility, self.average_access, self.accessArea)
        self.access_is_good = fuzz.interp_membership(self.accessibility, self.good_access, self.accessArea)
        

    def rules(self):
        """
        Define All rules and computerize the user's inputs for the fuzzy logic system 
        """
        self.rule1 = min(self.location_is_lessDemand, self.area_is_small, self.unfunishing)
        self.rule2 = min(self.location_is_lessDemand, max(self.area_is_small, self.area_is_average), self.access_is_good)
        self.rule3 = min(self.location_is_veryHighDemand, self.area_is_average, self.fac_is_low, self.access_is_average)
        self.rule4 = min(self.location_is_veryLessDemand, self.area_is_verysmall, self.fully_funishing)
        self.rule5 = min(self.location_is_lessDemand, self.fac_is_average, max(self.area_is_small, self.area_is_average))
        self.rule6 = min(max(self.location_is_lessDemand, self.location_is_averageDemand), self.access_is_good)
        self.rule7 = min(self.location_is_lessDemand, self.access_is_good, self.area_is_large, self.partially_funishing)
        self.rule8 = min(self.location_is_highDemand, self.access_is_good, max(self.bed_is_less, self.bath_is_average))
        self.rule9 = min(self.location_is_veryHighDemand, self.area_is_large, self.unfunishing)
        self.rule10 = min(self.access_is_good, self.area_is_average, (1 - self.unfunishing))
        self.rule11 = min(self.access_is_good, self.area_is_large, self.partially_funishing, self.bed_is_more, self.bath_is_more)
        


           

    def standardComposition_Min(self):
        """
        Define and computerize all the output for each rule
        """
        self.rulesList = []

        self.rulesList.append(np.fmin(self.rule1,self.below_price))
        self.rulesList.append(np.fmin(self.rule2,self.below_price))
        self.rulesList.append(np.fmin(self.rule3,self.below_price))
        self.rulesList.append(np.fmin(self.rule4,self.standard_price))
        self.rulesList.append(np.fmin(self.rule5,self.standard_price))
        self.rulesList.append(np.fmin(self.rule6,self.standard_price))
        self.rulesList.append(np.fmin(self.rule7,self.above_price))
        self.rulesList.append(np.fmin(self.rule8,self.above_price))
        self.rulesList.append(np.fmin(self.rule9,self.above_price))
        self.rulesList.append(np.fmin(self.rule10,self.high_price))
        self.rulesList.append(np.fmin(self.rule11,self.high_price))
        

    def standardComposition_Max(self):
        """
        used to find the final output in psf
        """
        temp = np.fmax(self.rulesList[0], self.rulesList[1])
        for r in self.rulesList[2:]:
            temp = np.fmax(temp, r)

        self.fuzzy_output = temp
    
    def defuzzification(self):
        """
        convert the fuzzy output to crisp output
        """
        self.price = fuzz.defuzz(self.psf,self.fuzzy_output, 'som')

    def graphDisplay(self):
        
        price_activation = fuzz.interp_membership(self.psf, self.fuzzy_output, self.price)
        psf0 = np.zeros_like(self.psf)
        fig, ax0 = plt.subplots(figsize=(8,3))
        ax0.plot(self.psf, self.below_price, 'b', linewidth=1.5, linestyle='--')
        ax0.plot(self.psf, self.standard_price, 'g', linewidth=1.5, linestyle='--')
        ax0.plot(self.psf, self.above_price, 'r', linewidth=1.5, linestyle='--')
        ax0.plot(self.psf, self.high_price, 'p', linewidth=1.5, linestyle='--')
        ax0.fill_between(self.psf,psf0, self.fuzzy_output, facecolor='Orange', alpha=0.5)
        ax0.plot([self.price, self.price], [0, price_activation],'k', linewidth=2.5, alpha=0.9)

        plt.show()

    def run(self):
        """
        this function is used to run all the function 
        and return the final output 
        which is the predicted psf price of the condominium 
        """
        self.membershipFunction()
        self.interpretingMF()
        self.rules()
        self.standardComposition_Min()
        self.standardComposition_Max()
        self.defuzzification()
def testData():                                                                                                                           
    col_names = ['Name', 'bedroom', 'bathroom', 
                'area size', 'psf', 'funishing', 
                'location median price (RM 1000)', 
                'facilities rating', 'accessibility rating']
    data = pd.read_csv('Data/Data Testing.csv', usecols=col_names, index_col=False)
    
    not_ok = 0
    predicted_price = []
    diff_price = []
    for i in range(len(data)):
        fuzzy_program = CondoPrice(data['location median price (RM 1000)'][i],
                                     data['bedroom'][i], 
                                     data['bathroom'][i], 
                                     data['funishing'][i], 
                                     data['area size'][i], 
                                     data['facilities rating'][i], 
                                     data['accessibility rating'][i])
        fuzzy_program.run()
        predicted_price.append("%.2f" % fuzzy_program.price)
        print(i, data['Name'][i])
        print("actual psf :", data['psf'][i])
        print("predict psf :", fuzzy_program.price)
        r = float(data['psf'][i]) - float(fuzzy_program.price)
        diff_price.append("%.2f" % abs(r))
        print("%.2f" % r)
        if(abs(r) <= 100):
            print("OK")
        else:
            print("not ok")
            not_ok+=1
        #fuzzy_program.graphDisplay()
    df = pd.DataFrame({'actual price': data['psf'],
                        'predicted price': predicted_price,
                        'difference price': diff_price})
    df.to_csv("output.csv", index=False)
    
def main():
    location = input("Enter the median price of the house location (in RM 1000): ")
    bedroom = input("Enter the number of bedroom : ")
    bathroom = input("Enter the number of bathroom : " )
    furnishing = input(" 0 : unfurnished \n 1 : partially furnished \n 2 : fully furnished \n Enter the furnishing status: ")
    areaSize = input("Enter the area size (in sqft) : ")
    facility = input("Enter the facilities rating between [0,10]: ")
    access = input("Enter the accessibility rating between [0,10]: ") 
    fuzzy_program = CondoPrice(int(location),
                                int(bedroom),
                                int(bathroom),
                                int(furnishing),
                                int(areaSize),
                                facility,
                                int(access))
    fuzzy_program.run()
    print(fuzzy_program.price)

if __name__ == '__main__':
    main()
    #testData()
    