import os
import json

class Location:
    def __init__(self, location_dict):
        self.facility = location_dict.get('facility', '')
        self.city = location_dict.get('city', '')
        self.state = location_dict.get('state', '')
        self.zip = location_dict.get('zip', '')
        self.country = location_dict.get('country', '')
        self.geo_point = location_dict.get('geoPoint', {})

    def __repr__(self):
        return (
            f"Location(\n"
            f"facility='{self.facility}', \n"
            f"city='{self.city}', \n"
            f"state='{self.state}', \n"
            f"zip='{self.zip}', \n"
            f"country='{self.country}', \n"
            f"geo_point={self.geo_point}\n"
            f")"
        )

class InclusionCriteria:
    def __init__(self, eligibility_module):
        self.criteria = eligibility_module.get("eligibilityCriteria", "")
        self.healthy_volunteers = eligibility_module.get("healthyVolunteers", "")
        self.sex = eligibility_module.get("sex", "")
        self.minimum_age = eligibility_module.get("minimumAge", "")

    def __repr__(self):
        return (
            f"InclusionCriteria(\n"
            f"criteria='{self.criteria[:50]}...', \n"
            f"healthy_volunteers='{self.healthy_volunteers}', \n"
            f"sex='{self.sex}', \n"
            f"minimum_age='{self.minimum_age}'\n"
            f")"
        )

class Intervention:
    def __init__(self, intervention_dict):
        self.type = intervention_dict.get('type', '')
        self.title = intervention_dict.get('name', '')
        self.description = intervention_dict.get('description', '')
        self.arm_group_labels = intervention_dict.get('armGroupLabels', [])
        self.other_names = intervention_dict.get('otherNames', [])

    def __repr__(self):
        return (
            f"Intervention(\n"
            f"type='{self.type}', \n"
            f"title='{self.title}', \n"
            f"description='{self.description[:50]}...', \n"
            f"arm_group_labels={self.arm_group_labels}, \n"
            f"other_names={self.other_names}\n"
            f")"
        )

class Endpoint:
    def __init__(self, endpoint_dict):
        self.title = endpoint_dict.get('measure', '')
        self.description = endpoint_dict.get('description', '')
        self.timeframes = endpoint_dict.get('timeFrame', '').split(", ")

    def __repr__(self):
        return (
            f"Endpoint(\n"
            f"title='{self.title}', \n"
            f"description='{self.description[:50]}...', \n"
            f"timeframes={self.timeframes}\n"
            f")"
        )


class Cohort:
    def __init__(self, group_dict, denoms):
        self.id = group_dict["id"]
        self.title = group_dict["title"]
        self.description = group_dict.get('description', '')
        self.units = denoms[0]['units']

        for denom in denoms[0]['counts']:
            if self.id == denom["groupId"]:
                self.denom = denom['value']
                break

    def __repr__(self):
        return (
            f"Cohort(\n"
            f"id='{self.id}', \n"
            f"title='{self.title}', \n"
            f"description='{self.description[:50]}...', \n"
            f"units='{self.units}', \n"
            f"denom={self.denom}\n"
            f")"
        )


class BaselineChar:
    def __init__(self, baseline_measure):
        self.title = baseline_measure.get('title', '')
        self.param_type = baseline_measure.get('paramType', '')
        self.dispersion_type = baseline_measure.get('dispersionType', '')
        self.unit_of_measure = baseline_measure.get('unitOfMeasure', '')
        self.measures = baseline_measure.get('classes', [{}])[0].get('categories', [{}])[0].get('measurements', [])
    
    def cohort_stats(self, cohort):
        for cohort_measure in self.measures:
            if cohort.id == cohort_measure["groupId"]:
                return cohort_measure

    def __repr__(self):
        return (
            f"BaselineChar(\n"
            f"title='{self.title}', \n"
            f"param_type='{self.param_type}', \n"
            f"dispersion_type='{self.dispersion_type}', \n"
            f"unit_of_measure='{self.unit_of_measure}', \n"
            f"measures_count={len(self.measures)}\n"
            f")"
        )


class OutcomeResult:
    def __init__(self, outcome_measure):
        self.title = outcome_measure["title"]
        self.type = outcome_measure.get('type', '')
        self.description = outcome_measure.get('description', '')
        self.pop_description = outcome_measure.get('populationDescription', '')
        self.reporting_status = outcome_measure.get('reportingStatus', '')
        self.param_type = outcome_measure.get('paramType', '')
        self.dispersion_type = outcome_measure.get('dispersionType', '')
        self.unit_of_measure = outcome_measure.get('unitOfMeasure', '')
        self.timeframes = outcome_measure.get('timeFrame', '').split(", ")
        self.measures = outcome_measure.get("classes", [])
        self.analyses = outcome_measure.get("analyses", [])
        denoms = outcome_measure.get("denoms", [])
        self.cohorts = [Cohort(cohort, denoms) for cohort in outcome_measure.get("groups", [])]

    def cohort_stats(self, cohort, title=''):
        for measure_class in self.measures:
            categories = measure_class.get('categories', [])
            # if measure_class.get('title', '') == title and len(categories) > 0:
            measurements = categories[0].get('measurements', [])
            for cohort_measure in measurements:
                if cohort.id == cohort_measure["groupId"]:
                    return cohort_measure

    def __repr__(self):
        return (
            f"OutcomeResult(\n"
            f"title='{self.title}', \n"
            f"type='{self.type}', \n"
            f"description='{self.description[:50]}...', \n"
            f"param_type='{self.param_type}', \n"
            f"timeframes={self.timeframes}, \n"
            f"measures_count={len(self.measures)}, \n"
            f"cohorts_count={len(self.cohorts)}\n"
            f")"
        )

    

class ClinicalTrial(object):

    def __init__(self, data_path, nct_id):
        self.data_path = data_path
        self.nct_id = nct_id 

        file_path = os.path.join(data_path, nct_id + ".json")
        with open(file_path, "r") as file:
            self.trial = json.load(file)
        
        self.protocol()
        self.results()

    def protocol(self):
        self.protocol = self.trial['protocolSection']
        self.brief_title = self.protocol.get("identificationModule", {}).get("briefTitle", "")
        self.official_title = self.protocol.get("identificationModule", {}).get("officialTitle", "")
        self.brief_summary = self.protocol.get("descriptionModule", {}).get("briefSummary", "")
        self.detailed_description = self.protocol.get("descriptionModule", {}).get("detailedDescription", "")
  
        self.results_first_posted = self.protocol.get("statusModule", {}).get("resultsFirstPostDateStruct", {}).get("date", "")

        self.conditions = self.protocol.get("conditionsModule", {}).get("conditions", [])
        self.keywords = self.protocol.get("conditionsModule", {}).get("keywords", [])
        self.study_type = self.protocol.get("designModule", {}).get("studyType", "")
        self.phases = self.protocol.get("designModule", {}).get("phases", [])
        self.total_enrolled = self.protocol.get("designModule", {}).get("enrollmentInfo", {}).get("count", "")
        self.alloc = self.protocol.get("designModule", {}).get('designInfo', {}).get('allocation', '')
    
        self.inclusion_criteria = InclusionCriteria(self.protocol.get("eligibilityModule", {}))
        self.interventions = [Intervention(intervention) for intervention in self.protocol.get("armsInterventionsModule", {}).get("interventions", [])]
        self.primary_endpoints = [Endpoint(endpoint) for endpoint in self.protocol.get("outcomesModule", {}).get("primaryOutcomes", [])]
        self.secondary_endpoints = [Endpoint(endpoint) for endpoint in self.protocol.get("outcomesModule", {}).get("secondaryOutcomes", [])]
        self.locations = [Location(location) for location in self.protocol.get("contactsLocationsModule", {}).get("locations", [])]

        self.duration = self.trial.get("duration", None)

    def results(self):
        self.results = self.trial['resultsSection']
        denoms = self.results.get("baselineCharacteristicsModule", {}).get("denoms", [])
        self.bg_cohorts = [Cohort(cohort, denoms) for cohort in self.results.get("baselineCharacteristicsModule", {}).get("groups", [])]
        self.baseline_char = [BaselineChar(measure) for measure in self.results.get("baselineCharacteristicsModule", {}).get("measures", [])]
        self.outcome_results = [OutcomeResult(outcome) for outcome in self.results.get("outcomeMeasuresModule", {}).get("outcomeMeasures", []) if outcome.get("type") == "PRIMARY"]
    
    def __repr__(self):
        return (
            f"ClinicalTrial(\n"
            f"nct_id='{self.nct_id}', \n"
            f"brief_title='{self.brief_title[:50]}...', \n"
            f"study_type='{self.study_type}', \n"
            f"phases={self.phases}, \n"
            f"results_first_posted='{self.results_first_posted}', \n"
            f"total_enrolled={self.total_enrolled}, \n"
            f"alloc='{self.alloc}', \n"
            f"conditions={self.conditions}, \n"
            f"keywords={self.keywords}, \n"
            f"interventions={[i.title for i in self.interventions]}, \n"
            f"primary_endpoints={[e.title for e in self.primary_endpoints]}\n"
            f")"
        )
