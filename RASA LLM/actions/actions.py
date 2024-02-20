from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
from rasa_sdk import Tracker
from rasa_sdk.forms import FormAction

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

        response_dict = {
            (True, True): f"Hello, {name}! It's fascinating that you reside in {place}. How can I assist you further?",
            (True, False): f"Hello, {name}! I'd love to learn more about where you're currently located. Could you share your place with me?",
            (False, True): f"It seems like you're in {place}! What's your name, if you don't mind me asking?",
            (False, False): "It looks like I didn't catch both your name and place. Could you please provide them again?"
        }

        message = response_dict.get((bool(name), bool(place)), "Unexpected scenario!")

        dispatcher.utter_message(text=message)
        return []


class AppointmentForm(FormAction):
    def name(self) -> Text:
        return "appointment_form"

    @staticmethod
    def required_slots(tracker: Tracker) -> List[Text]:
        return ["name", "place", "appointment_date"]

    def slot_mappings(self) -> Dict[Text, Any]:
        return {
            "name": self.from_entity(entity="name"),
            "place": self.from_entity(entity="place"),
            "appointment_date": self.from_entity(entity="datetime")
        }

    def submit(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        name = tracker.get_slot("name")
        place = tracker.get_slot("place")
        appointment_date = tracker.get_slot("appointment_date")

        # Perform any actions or validations here based on the collected information
        # For example, you can save the appointment details to a database

        dispatcher.utter_message(
            text=f"Great! Your appointment is scheduled. Name: {name}, Place: {place}, Date: {appointment_date}"
        )

        return []
