from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List

class ActionSayPhone(Action):

    def name(self) -> Text:
        return "action_say_phone"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        phone = tracker.get_slot("phone")
        if not phone:
            dispatcher.utter_message(text="Sorry, I don't know your phone number.")
        else:
            dispatcher.utter_message(text=f"Your phone number is {phone}")
        return []

class UtterGoodbye(Action):
    def name(self) -> Text:
        return "utter_goodbye"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Goodbye!")
        return []



class ActionBookAppointment(Action):

    def name(self) -> Text:
        return "action_book_appointment"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Implement logic to handle appointment booking
        dispatcher.utter_message(text="Sure, I can help you with that. When would you like to schedule an appointment?")
        return []


class ActionMedicalInfo(Action):

    def name(self) -> Text:
        return "action_medical_info"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Implement logic to provide medical information
        dispatcher.utter_message(text="I can provide information about medical conditions, medications, and procedures. What are you looking for?")
        return []


class ActionPrescription(Action):

    def name(self) -> Text:
        return "action_prescription"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Implement logic to handle prescription requests
        dispatcher.utter_message(text="To prescribe medication, I need more details about your symptoms. Can you describe them?")
        return []


class ActionInsuranceInfo(Action):

    def name(self) -> Text:
        return "action_insurance_info"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Implement logic to provide information about health insurance
        dispatcher.utter_message(text="For information about your health insurance, please provide your policy details.")
        return []

class ActionSayNameAndPlace(Action):

    def name(self) -> Text:
        return "action_say_name_and_place"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        name = tracker.get_slot("name")
        place = tracker.get_slot("place")

        if name and place:
            dispatcher.utter_message(text=f"Nice to meet you, {name}! Do you enjoy being in {place}?")
        else:
            dispatcher.utter_message(text="I didn't catch your name or place. Can you please provide them again?")
        return []
