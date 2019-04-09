"""

This file set ups the health system resources for each district.

It creates one facility of each type for each district, and allows the 'Referral Hospitals' to connect to all the
districts in a region,  and the 'National' Hospital' to connect to all districts in the country.

"""

# CHAI DATA SET:
workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ORIGINAL_Optimization model import_Malawi_20180315 v10.xlsx'

# OUTPUT RESOURCE_FILES TO:
resourcefilepath='/Users/tbh03/PycharmProjects/TLOmodel/resources/'


# ----------
# ----------
# ----------

import pandas as pd
import numpy as np

# ----------

# Import all of the 'CurrentStaff' sheet
wb_import= pd.read_excel(workingfile,sheet_name='CurrentStaff',header=None)

# ----------

# Make dataframe summarising the officer types and the officer codes:
officer_types_table=wb_import.loc[2:3,64:84].transpose().reset_index(drop=True).copy()
officer_types_table.columns=['Officer_Type','Officer_Type_Code']
officer_types_table.to_csv(resourcefilepath + 'ResourceFile_Officer_Types_Table.csv')

# ----------

# Extract just the section about "Funded TOTAl Staff'
wb_extract = wb_import.loc[3:37,64:84]
wb_extract=wb_extract.drop([4,5])
wb_extract.columns = wb_extract.iloc[0]
wb_extract=wb_extract.drop([3])
wb_extract=wb_extract.reset_index(drop=True)
wb_extract.fillna(0, inplace=True) # replace all null values with zero values

# Add in the column to the dataframe for the labels districts (and key hospitals):
labels= wb_import.loc[6:37,0].reset_index(drop=True)
is_distlevel = labels.copy()
is_distlevel[0:27]=True
is_distlevel[27:]=False

wb_extract.loc[:,'District_Or_Hospital'] = labels
wb_extract.loc[:,'Is_DistrictLevel']=is_distlevel

# Finished import from the CHAI excel:
staffing_table = wb_extract

# Sort out which are district allocations and which are central hospitals
staffing_table .loc[staffing_table['District_Or_Hospital']=='HQ or missing','District_Or_Hospital'] = 'National Hospital'
staffing_table .loc[staffing_table['District_Or_Hospital']=='KCH','District_Or_Hospital'] = 'Referral Hospital_Central'
staffing_table .loc[staffing_table['District_Or_Hospital']=='MCH','District_Or_Hospital'] = 'Referral Hospital_Northern'
staffing_table .loc[staffing_table['District_Or_Hospital']=='QECH','District_Or_Hospital'] = 'Referral Hospital_Southern'

# Put the ZCH (assume Zomba City Hospital) into the Zomba district level allocation
staffing_table .loc[staffing_table ['District_Or_Hospital']=='ZCH','District_Or_Hospital'] = 'Zomba'
staffing_table=pd.DataFrame(staffing_table.groupby(by=['District_Or_Hospital']).sum()).reset_index()


# The following districts are not in the CHAI data because they are included within other districts.
# For now, we will say thay the division beween these cities and the wide district (in which they are included) is equal.

# Add in Likoma (part Nkhata Bay)
# Add in Lilongwe City (part of Lilongwe)
# Add in Mzuzu City (part of Mziba) ASSUMED
# Add in Zomba City (part of Zomba)
# Add in Blantyre City (part of Blantyre)

# create mapping: the new districts : super_district
split_districts= (
    ('Likoma', 'Nkhata Bay'),
    ('Lilongwe City', 'Lilongwe'),
    ('Mzuzu City', 'Mzimba'),
    ('Zomba City','Zomba'),
    ('Blantyre City', 'Blantyre')
)

for i in np.arange(0,len(split_districts)):
    new_district = split_districts[i][0]
    super_district = split_districts[i][1]

    record = staffing_table.iloc[0].copy() # get a row of the staffing table

    # make a the record for the new district
    record['District_Or_Hospital']=new_district
    record['Is_DistrictLevel']=True

    # get total staff level from the super districts
    cols = set(staffing_table.columns).intersection(set(officer_types_table.Officer_Type_Code))

    total_staff = staffing_table.loc[staffing_table['District_Or_Hospital']==super_district, cols].values.squeeze()

    record.loc[cols]= 0.5*total_staff # assign half the staff to the new district
    staffing_table =staffing_table.append(record).reset_index(drop=True)

    # take staff away from the super district
    staffing_table.loc[staffing_table['District_Or_Hospital']==super_district,cols]= staffing_table.loc[staffing_table['District_Or_Hospital']==super_district,cols] - record.loc[cols]


# Confirm the merging will be perfect:
pop = pd.read_csv(resourcefilepath+ 'ResourceFile_District_Population_Data.csv')

assert set(pop['District'].values) == set(staffing_table.loc[staffing_table['Is_DistrictLevel'],'District_Or_Hospital'])
assert len(pop['District'].values) == len(staffing_table.loc[staffing_table['Is_DistrictLevel'],'District_Or_Hospital'])

# ... double check by doing the merge explicitly
pop_districts = pd.DataFrame({'District': pd.unique(pop['District'])})
chai_districts = pd.DataFrame({'District': staffing_table.loc[staffing_table['Is_DistrictLevel'],'District_Or_Hospital']})

merge_result=pop_districts.merge(chai_districts,how='inner',indicator=True)
assert all(merge_result['_merge']=='both')
assert len(merge_result) == len(pop_districts)


# --------------

# Flip from wide to long format
staffing_table.loc[:,staffing_table.columns[1:]]=staffing_table.loc[:,staffing_table.columns[1:]].round(0).astype(int) # Make values integers (after rounding)

staff_list=pd.melt(staffing_table, id_vars=['District_Or_Hospital','Is_DistrictLevel'], var_name='Officer_Type_Code', value_name='Number')

# Repeat rows so that it is one row per staff member
staff_list['Number']=staff_list['Number'].astype(int)
staff_list = staff_list.loc[staff_list.index.repeat(staff_list['Number'])]
staff_list = staff_list.reset_index(drop=True)
staff_list=staff_list.drop(['Number'],axis=1)

# check that the number of rows in this staff_list is equal to the total number of staff from the input data
assert len(staff_list) == staffing_table.iloc[:,1:-1].sum().sum()

staff_list['Is_DistrictLevel']= staff_list['Is_DistrictLevel'].astype(bool)

# assign an arbitary staff_id
staff_list['Staff_ID']=staff_list.index

#  --------

# *** Create the Master Facilities List
# This will be a listing of each facility and the district(s) to which they attach
# We assume that the set of facilities of one type operate as one big facility type within the district.

# declare the Facility_Type variable
Facility_Types = ['Community Health Station', 'Health Centre', 'Non-District Hospital', 'District Hospital', 'Referral Hospital', 'National Hospital']
Facility_Levels = [0,1,2,3,4,5]
Facility_Types_Levels = dict(zip(Facility_Types,Facility_Levels))

# Create empty dataframe that will be the Master Facilities List (mfl)
mfl = pd.DataFrame(columns= ['Facility_Type','Facility_Level','District','Region'])

pop_districts = pop['District'].values
pop_regions = pd.unique(pop['Region'])

for d in pop_districts:
    df = pd.DataFrame({'Facility_Type': Facility_Types[0:4], 'Facility_Level' : [0, 1, 2, 3], 'District': d, 'Region': pop.loc[pop['District']==d,'Region'].values[0] })
    mfl= mfl.append(df,ignore_index=True,sort=True)

# Add in the Referral Hospitals, one for each region
for r in pop_regions:
    mfl=mfl.append(pd.DataFrame({
        'Facility_Type': Facility_Types[4], 'Facility_Level' : [4], 'District': None, 'Region': r
    }),ignore_index=True,sort=True)

# Add in the National Hosital, one for whole country
mfl=mfl.append(pd.DataFrame({
    'Facility_Type': Facility_Types[5], 'Facility_Level' : [5], 'District': None, 'Region': None
}),ignore_index=True,sort=True)

# Create the Facility_ID
mfl.loc[:,'Facility_ID']=mfl.index

# Create a unique name for each Facility
name=mfl['Facility_Type'] + '_' + mfl['District']
name.loc[mfl['Facility_Type']=='Referral Hospital'] ='Referral Hospital' + '_' + mfl.loc[mfl['Facility_Type']=='Referral Hospital','Region']
name.loc[mfl['Facility_Type']=='National Hospital'] ='National Hospital'

mfl.loc[:,'Facility_Name']=name

mfl.to_csv(resourcefilepath + 'ResourceFile_Master_Facilities_List.csv')

#  --------

# Create a simple mapping of all the facilities that persons in a district can access
facilities_by_district=pd.DataFrame(columns=mfl.columns)

for d in pop_districts:
    the_region = pop.loc[pop['District']==d,'Region'].copy().values[0]
    district_facs = mfl.loc[mfl['District']==d]

    region_fac = mfl.loc[ pd.isnull(mfl['District']) & (mfl['Region']==the_region) ].copy().reset_index(drop=True)
    region_fac.loc[0,'District']=d

    national_fac = mfl.loc[ pd.isnull(mfl['District']) & pd.isnull(mfl['Region']) ].copy().reset_index(drop=True)
    national_fac.loc[0,'District'] = d
    national_fac.loc[0, 'Region'] = the_region

    facilities_by_district = pd.concat([facilities_by_district , district_facs,region_fac, national_fac],ignore_index=True)

assert len(facilities_by_district) == len(pop_districts) * len(Facility_Types)

facilities_by_district.to_csv(resourcefilepath + 'ResourceFile_Facilities_For_Each_District.csv')

#  --------

# Create a mapping of how these different officer types will go to particular facility types
# The level is important as this determines how many other people are drawing on their time.

Facility_By_Officer=pd.DataFrame({'Officer_Type': officer_types_table.Officer_Type, 'Officer_Type_Code':officer_types_table.Officer_Type_Code ,'Facility_Type_Can_Work_In':None})

Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Medical Officer / Specialist','Facility_Type_Can_Work_In']={'District Hospital','Non-District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Clinical Officer / Technician','Facility_Type_Can_Work_In']={'District Hospital','Non-District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Med. Assistant','Facility_Type_Can_Work_In']={'District Hospital','Non-District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Nurse Officer','Facility_Type_Can_Work_In']={'District Hospital', 'Community Health Station','Non-District Hospital', 'Health Centre', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Nurse Midwife Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Non-District Hospital','Community Health Station', 'Health Centre', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Pharmacist','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre', 'Non-District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Pharm Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre', 'Non-District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Pharm Assistant','Facility_Type_Can_Work_In']={'District Hospital', 'Health Centre','Non-District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Lab Officer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','Non-District Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Lab Technician','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','Non-District Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Lab Assistant','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','Non-District Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='DCSA','Facility_Type_Can_Work_In']={'Community Health Station'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Dental Officer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Dental Therapist','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Dental Assistant','Facility_Type_Can_Work_In']={'District Hospital','Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Mental Health Staff','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Nutrition Staff','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','National Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Radiographer','Facility_Type_Can_Work_In']={'National Hospital', 'Referral Hospital','District Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Radiography Technician','Facility_Type_Can_Work_In']={'National Hospital', 'Referral Hospital','District Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Sonographer','Facility_Type_Can_Work_In']={'District Hospital', 'Referral Hospital','District Hospital'}
Facility_By_Officer.at[Facility_By_Officer['Officer_Type']=='Radiotherapy Technician','Facility_Type_Can_Work_In']={'National Hospital', 'Referral Hospital','District Hospital'}


# check that have only ever used correct Facility_Type labels and check that each one has a type including at least one district level
for i in np.arange(0, len(Facility_By_Officer)):
    x = Facility_By_Officer['Facility_Type_Can_Work_In'][i]
    assert x.issubset(set(Facility_Types))
    assert len(x.intersection(set(Facility_Types[0:4])))>0

# check that every Officer_Type has been covered
assert set(Facility_By_Officer['Officer_Type_Code']) == set(officer_types_table.Officer_Type_Code)



#  --------


# *** Now look at the types of appointments and the draw on officer's time associated with each

sheet= pd.read_excel(workingfile,sheet_name='Time_Base',header=None)


# ----------


# get rid of the junky rows
trimmed=sheet.loc[[7,8,9,11,12,14,15,17,18,20,21,23,24,26,27],]
data_import=pd.DataFrame(data=trimmed.iloc[1:,2:].values,columns=trimmed.iloc[0,2:],index=trimmed.iloc[1:,1])

data_import=data_import.dropna(axis='columns',how='all') # get rid of the 'spacer' columns
data_import=data_import.fillna(0)

# get rid of records for which there is no call on time of any type of officer
data_import=data_import.drop(columns=data_import.columns[data_import.sum()==0])

# We note that the DCSA (CHW) never has a time requirement and that no appointments can be serviced at the HealthPost.
# We remedy this by inserting a new type of appoitment, which only the DCSA can service, and the time taken is 10 minutes.
new_appt_for_CHW=pd.Series(index=data_import.index,
                           name='E01_ConWithDCSA',  # New appointment type is a consultation with the DCSA (community health worker)
                           data=[
                                    0,                          # Central Hosp - Time
                                    0,                          # Central Hosp - Percent
                                    0,                          # District Hosp - Time
                                    0,                          # District Hosp - Percent
                                    0,                          # Comm Hosp - Time
                                    0,                          # Comm Hosp - Percent
                                    0,                          # Urban Health Centre - Time
                                    0,                          # Urban Health Centre - Percent
                                    0,                          # Rural Health Centre - Time
                                    0,                          # Rural Health Centre - Percent
                                    10.0,                       # Health Post - Time
                                    1.0,                        # Health Post - Percent
                                    0,                          # Dispensary - Time
                                    0,                          # Dispensary - Percent
])

data_import=pd.concat([data_import,new_appt_for_CHW],axis=1)


# break apart composite to give the appt_type and the officer_type
# This is used to know which column to read below...
chai_composite_code=pd.Series(data_import.columns)
chai_code=chai_composite_code.str.split(pat='_',expand=True).reset_index(drop=True)
chai_code=chai_code.rename(columns={0:'Officer_Type_Code',1:'Appt_Type_Code'})

# check that officer codes line up with the officer codes already imported
assert set(chai_code['Officer_Type_Code']).issubset(set(officer_types_table['Officer_Type_Code']))


# ----------
# Make dataframe summarising the types of appointments

retained_appt_type_code = pd.unique(chai_code['Appt_Type_Code'])

appt_types_table_import=sheet.loc[(1,2,6),2:].transpose().reset_index(drop=True).copy()
appt_types_table_import=appt_types_table_import.rename(columns={1:'Appt_Cat',2:'Appt_Type',6:'Appt_Type_Code'})
appt_types_table_import['Appt_Cat']=pd.Series(appt_types_table_import['Appt_Cat']).fillna(method='ffill')
appt_types_table_import['Appt_Type']=pd.Series(appt_types_table_import['Appt_Type']).fillna(method='ffill')
appt_types_table_import['Appt_Type_Code']=pd.Series(appt_types_table_import['Appt_Type_Code']).fillna(method='ffill')
appt_types_table_import=appt_types_table_import.drop_duplicates().reset_index(drop=True)

# starting with the retained appt codes, merge in these descriptions
appt_types_table= pd.DataFrame(data={'Appt_Type_Code':retained_appt_type_code}).merge(appt_types_table_import,on='Appt_Type_Code',how='left',indicator=True)

# Fill in the missing information about the appointment type that was added above
appt_types_table.loc[appt_types_table['Appt_Type_Code']==new_appt_for_CHW.name.split('_')[1],'Appt_Cat']=new_appt_for_CHW.name.split('_')[1]
appt_types_table.loc[appt_types_table['Appt_Type_Code']==new_appt_for_CHW.name.split('_')[1],'Appt_Type']=new_appt_for_CHW.name.split('_')[1]

# drop the merge check column
appt_types_table.drop(columns='_merge',inplace=True)

# Check no holes
assert not pd.isnull(appt_types_table).any().any()

appt_types_table.to_csv(resourcefilepath + 'ResourceFile_Appt_Types_Table.csv')
# ----------

# Now, Make the ApptTimeTable
# (Table that gives for each appointtment, when occuring in each appt_type at each facility type, the time of each type of officer required

# The sheet gives the % of appointments that require a particular type of officer and the time taken if it does
# So, turn that into an Expectation of the time taken for each type of officer (multiplying together)

# This sheet distinguished between different types of facility. We will map these to the facility types that have been defined.
# In doing this, we ignore the distinction made here between urban and rural health centres, and just use the rural.

# CHAI: Central_Hospital ---> our "Referral Hospital" and "National Hospital"
# CHAI: District_Hospital ---> our "District Hospital"
# CHAI: Community_Hospital ---> our "Non-District Hospital"
# CHAI: Urban_HealthCentre ---> (ignored)
# CHAI: Rural_HealthCentre ---> our "Health Centre"
# CHAI: HealthPost ---> our "Community Health Station"
# CHAI: Dispensary ---> (ignored)


Central_Hospital_ExpecTime = data_import.loc['CenHos'] * data_import.loc['CenHos_Per']
District_Hospital_ExpecTime = data_import.loc['DisHos'] * data_import.loc['DisHos_Per']
Community_Hospital_ExpecTime = data_import.loc['ComHos'] * data_import.loc['ComHos_Per']
Rural_HealthCentre_ExpecTime = data_import.loc['RurHC'] * data_import.loc['RurHC_Per']
HealthPost_ExpecTime = data_import.loc['HP'] * data_import.loc['HP_Per']

X=pd.DataFrame({
            'National Hospital': Central_Hospital_ExpecTime,
            'Referral Hospital': Central_Hospital_ExpecTime,
            'District Hospital': District_Hospital_ExpecTime,
            'Non-District Hospital': Community_Hospital_ExpecTime,
            'Health Centre': Rural_HealthCentre_ExpecTime,
            'Community Health Station' : HealthPost_ExpecTime
              })

# split out the index into appointment type and officer type
labels=pd.Series(X.index,index=X.index).str.split(pat='_',expand=True)
labels=labels.rename(columns={0:'Officer_Type_Code', 1:'Appt_Type_Code'})
Y=pd.concat([X,labels],axis=1)
ApptTimeTable=pd.melt(Y,id_vars=['Officer_Type_Code','Appt_Type_Code'],value_vars=Facility_Types,var_name='Facility_Type',value_name='Time_Taken')

# Merge in Officer_Type and Facility_Level
ApptTimeTable=ApptTimeTable.merge(officer_types_table,on='Officer_Type_Code')
ApptTimeTable.loc[:,'Facility_Level']=ApptTimeTable['Facility_Type'].map(Facility_Types_Levels)

# confirm that we have the same number of entries as we were expecting
assert len(ApptTimeTable) == len(Facility_Types)*len(data_import.columns)

# drop the rows that contain no call on resources
ApptTimeTable=ApptTimeTable.drop(ApptTimeTable[ApptTimeTable['Time_Taken']==0].index)

# Save
ApptTimeTable.to_csv(resourcefilepath + 'ResourceFile_Appt_Time_Table.csv')


# -------
# -------
# -------

# Look to see where different types of staff member need to be located:
# This is just a reverse reading of where there are non-zero requests for time of particular officer-types

Officers_Need_For_Appt=pd.DataFrame(columns=['Facility_Type','Appt_Type_Code','Officer_Type_Codes'])

for a in appt_types_table['Appt_Type_Code'].values:
    for f in Facility_Types:

        # get the staff types required for this appt

        block=ApptTimeTable.loc[ (ApptTimeTable['Appt_Type_Code']==a) & (ApptTimeTable['Facility_Type']==f)]

        if len(block)==0:
            # no requirement expressed => The appt is not possible at this location
            Officers_Need_For_Appt=Officers_Need_For_Appt.append(
                                                                {'Facility_Type': f,
                                                                 'Appt_Type_Code': a,
                                                                 'Officer_Type_Codes': False
                                                                 },ignore_index=True)

        else:
            need_officer_types=list(block['Officer_Type_Code'])
            Officers_Need_For_Appt=Officers_Need_For_Appt.append(
                                                                {'Facility_Type': f,
                                                                 'Appt_Type_Code': a,
                                                                 'Officer_Type_Codes': need_officer_types
                                                                 },ignore_index=True)

# Turn this into the the set of staff that are required for each type of appointment
Fac_By_Officer= pd.DataFrame(columns = Facility_Types,index =officer_types_table['Officer_Type_Code'].values )
Fac_By_Officer=X.fillna(False)

for o in officer_types_table['Officer_Type_Code'].values:

    for i in Officers_Need_For_Appt.index:
        fac= Officers_Need_For_Appt.loc[i].Facility_Type
        officer_types = Officers_Need_For_Appt.loc[i].Officer_Type_Codes

        if officer_types != False:

            if (o in officer_types):
                Fac_By_Officer.loc[
                    ( Fac_By_Officer.index==o ),
                fac]=True


# We note that two officer_types ("T01: Nutrition Staff", "R03: Sonographer" and "RO4: Radiotherapy technican") are apparently not called by any appointment type

# Assign that the Nutrition Staff will go to the District, Referral and National Hospitals
Fac_By_Officer.loc['T01',['District Hospital','Referral Hospital','National Hospital']]=True

# Assign that the Sonographer will go to the District, Referral and National Hospitals
Fac_By_Officer.loc['R03',['District Hospital','Referral Hospital','National Hospital']]=True

# Assign that the Radiotherapist will go to the National Hospital
Fac_By_Officer.loc['R04','National Hospital']=True

# Check that all types of officer are allocated to at least one type of facility
assert (Fac_By_Officer.sum(axis=1)>0).all()

#-----------------
#-----------------
#-----------------



# *** Determine how staff are allocated to the facilities

# Create pd.Series for holding the assignments between staff member (in staff_list) and facility (in mfl)
facility_assignment =pd.DataFrame(index=staff_list.index,columns=['Staff_ID','Facility_ID'])
facility_assignment['Staff_ID']=staff_list['Staff_ID']

# Loop through each staff member and allocate them to
for staffmember in staff_list.index:

    officer =staff_list.loc[staffmember].Officer_Type_Code

    if staff_list.at[staffmember,'Is_DistrictLevel']:
        # This staff member is allocated to a district
        district= staff_list.at[staffmember,'District_Or_Hospital']

        # TODO: get this from Fac_By_Officer BUT MAKE SURE TO EXCLUDE REFERRAL AND NATIONAL
        possible_fac_types= list((Facility_By_Officer.loc[Facility_By_Officer['Officer_Type_Code']==officer,'Facility_Type_Can_Work_In']).iloc[0])

        # Get the facilities to which this staff member might be allocated
        suitable_facilities=mfl.loc[ (mfl['Facility_Type'].isin(possible_fac_types) ) & ( mfl['District'] == district ) ]

        assert len(suitable_facilities)>0

        # if there is a suitable facility for this office to go into
        x=np.random.choice(len(suitable_facilities))
        assigned_facility_id=suitable_facilities.iloc[x].Facility_ID

    else:
        # This staff member is allocated to one of the above-district types of facilities

        # TODO: confirm that this accord with Fac_By_Officer; LIMIT TO REFERRAL AND NATIONAL (I.E. NO E01)

        hospital = staff_list.at[staffmember,'District_Or_Hospital']

        if hospital == 'National Hospital':
            # Find the facility_id that corresponds to this hospital
            assigned_facility_id =mfl.loc[mfl['Facility_Name']==hospital,'Facility_ID'].values[0]

        else:
            # It's a referral hospital
            x = hospital.split('_')
            region=x[1]
            assigned_facility_id =mfl.loc[
                                        (mfl['Facility_Type']=='Referral Hospital') & (mfl['Region']==region)
                                    ,'Facility_ID'].values[0]



    # check the assigned_facility_id is a real one (i.e. not empty)
    assert set([assigned_facility_id]).issubset(set(mfl['Facility_ID']))

    # make the assignment
    facility_assignment.at[staffmember,'Facility_ID'] = assigned_facility_id



#-----------------
#-----------------
#-----------------

# *** Get Hours Worked Per Staff Member
# Reading-in the number of working hours and days for each type of officer

sheet= pd.read_excel(workingfile,sheet_name='PFT',header=None)
officer_types_import=sheet.iloc[2,np.arange(2,23)]

assert set(officer_types_import) == set( officer_types_table['Officer_Type_Code'] )
assert len(officer_types_import) == len( officer_types_table['Officer_Type_Code'] )

hours=sheet.iloc[38,np.arange(2,23)]

days_per_year_men=sheet.iloc[15,np.arange(2,23)]
days_per_year_women=sheet.iloc[16,np.arange(2,23)]
days_per_year_pregwomen=sheet.iloc[17,np.arange(2,23)]

fr_men=sheet.iloc[53,np.arange(2,23)]
fr_women=sheet.iloc[55,np.arange(2,23)] - sheet.iloc[57,np.arange(2,23)] # taking off pregnant women: this is for non-pregnant women
fr_pregwomen= sheet.iloc[57,np.arange(2,23)]

workingdays= ( fr_men*days_per_year_men ) + ( fr_women*days_per_year_women ) + ( fr_pregwomen*days_per_year_pregwomen )

minutes_per_day = hours * 60

minutes_per_year = minutes_per_day * workingdays

# What we will use is the average number of minutes worked per day:
av_minutes_per_day = minutes_per_year / 365.25

patient_facing_time = pd.DataFrame({'Officer_Type_Code': officer_types_import ,'Working_Days_Per_Year':workingdays,'Hours_Per_Day':hours, 'Av_Minutes_Per_Day':av_minutes_per_day }).reset_index(drop=True)


#-----------------
#-----------------
#-----------------


# --- Create final table of daily time available at each facilty by officer type: Facility_ID, Facility_Type, Facility_Level, Officer_Type, Officer_Typpe_Code, Total Average Minutes Per Day

# merge in officer type
X=staff_list.merge(facility_assignment, on='Staff_ID',how='left')
X=X.drop(columns=['District_Or_Hospital','Is_DistrictLevel'])

# merge in time that each officer type can spent on appointments
X= X.merge(patient_facing_time,on='Officer_Type_Code',how='left')
X= X.drop(columns=['Working_Days_Per_Year','Hours_Per_Day'])

# Now collapse across the staff_ID in order to give a summary per facility type and officer type: summing av minutes per day
Y = pd.DataFrame(X.groupby(['Facility_ID','Officer_Type_Code'])[['Av_Minutes_Per_Day']].sum()).reset_index()
Y = Y.rename(columns={'Av_Minutes_Per_Day':'Total_Minutes_Per_Day'})

# Merge in information about the facility:
Y = Y.merge(mfl,on='Facility_ID',how='left')

# Merge in the officer code name
Y = Y.merge(officer_types_table,on='Officer_Type_Code',how='left')

Y.to_csv(resourcefilepath+'ResourceFile_Daily_Capabilities.csv')


#-----------------
#-----------------
#-----------------
