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

        response_dict = {
            (True, True): f"Hello, {name}! It's fascinating that you reside in {place}. How can I assist you further?",
            (True, False): f"Hello, {name}! I'd love to learn more about where you're currently located. Could you share your place with me?",
            (False, True): f"It seems like you're in {place}! What's your name, if you don't mind me asking?",
            (False, False): "It looks like I didn't catch both your name and place. Could you please provide them again?"
        }

        message = response_dict.get((bool(name), bool(place)), "Unexpected scenario!")

        dispatcher.utter_message(text=message)
        return []
# from rasa_sdk.forms import FormAction
# from rasa_sdk import Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from typing import Dict, Any, List, Text

# class AppointmentForm(FormAction):
#     def name(self) -> Text:
#         return "appointment_form"

#     @staticmethod
#     def required_slots(tracker: Tracker) -> List[Text]:
#         return ["name", "place", "appointment_date"]

#     def slot_mappings(self) -> Dict[Text, Any]:
#         return {
#             "name": [self.from_text()],
#             "place": [self.from_text()],
#             "appointment_date": [self.from_entity(entity="date")]
#         }

#     def submit(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: Dict[Text, Any],
#     ) -> List[Dict[Text, Any]]:
#         # Implement logic to handle the appointment booking with the collected information
#         dispatcher.utter_message(text="Great! Your appointment has been scheduled.")
#         return []



from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

ALLOWED_PIZZA_SIZES = ["small", "medium", "large", "extra-large", "extra large", "s", "m", "l", "xl"]
ALLOWED_PIZZA_TYPES = ["mozzarella", "fungi", "veggie", "pepperoni", "hawaii"]

class ValidateSimplePizzaForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_simple_pizza_form"

    def validate_pizza_size(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `pizza_size` value."""

        if slot_value.lower() not in ALLOWED_PIZZA_SIZES:
            dispatcher.utter_message(text=f"We only accept pizza sizes: s/m/l/xl.")
            return {"pizza_size": None}
        dispatcher.utter_message(text=f"OK! You want to have a {slot_value} pizza.")
        return {"pizza_size": slot_value}

    def validate_pizza_type(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `pizza_type` value."""

        if slot_value not in ALLOWED_PIZZA_TYPES:
            dispatcher.utter_message(text=f"I don't recognize that pizza. We serve {'/'.join(ALLOWED_PIZZA_TYPES)}.")
            return {"pizza_type": None}
        dispatcher.utter_message(text=f"OK! You want to have a {slot_value} pizza.")
        return {"pizza_type": slot_value}