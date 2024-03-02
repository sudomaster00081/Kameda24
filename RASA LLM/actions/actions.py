from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.events import EventType
from rasa_sdk.types import DomainDict
from typing import Dict, Text, Any, List, Union


import os
from typing import Any, Text, Dict, List
import pandas as pd
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import openai 
import json

class RestaurantAPI(object):

    def __init__(self):
        self.db = pd.read_csv("restaurants.csv")

    def fetch_restaurants(self):
        return self.db.head()

    def format_restaurants(self, df, header=True) -> Text:
        return df.to_csv(index=False, header=header)


class ChatGPT(object):

    def __init__(self):
        self.url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
        self.headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('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')}"
        }
        self.prompt = "Answer the following question, based on the data shown. " \
            "Answer in a complete sentence and don't say anything else."

    def ask(self, restaurants, question):
        content  = self.prompt + "\n\n" + restaurants + "\n\n" + question
        body = {
            "model":self.model, 
            "messages":[{"role": "user", "content": content}]
        }
        result = requests.post(
            url=self.url,
            headers=self.headers,
            json=body,
        )
        return result.json()["choices"][0]["message"]["content"]
    

def ask_distance(restaurant_list):
    content = "measure the least distance with each given restaurant" +'/n/n' + restaurant_list
    completion = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[{"role": "user", "content": content}],
    functions=[
    {
        "name": "get_measure",
        "description": "Get the least distance",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "list of all the restaurants and distances as a dictionary(restuarant_name:distance)",
                },
            },
            "required": ["distance"],
        },
    }
        ],
        function_call={"name":"get_measure"}
    )
    return completion.choices[0].message



restaurant_api = RestaurantAPI()
chatGPT = ChatGPT()

class ActionShowRestaurants(Action):

    def name(self) -> Text:
        return "action_show_restaurants"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        restaurants = restaurant_api.fetch_restaurants()
        results = restaurant_api.format_restaurants(restaurants)
        readable = restaurant_api.format_restaurants(restaurants[['Restaurants', 'Rating']], header=False)
        dispatcher.utter_message(text=f"Here are some restaurants:\n\n{readable}")

        return [SlotSet("results", results)]


def get_distance(d):
    d = json.loads(d)
    for i in d.keys():
        d[i]= float(d[i])
    t = min(d, key =d.get)
    return t

class ActionRestaurantsDetail(Action):
    def name(self) -> Text:
        return "action_restaurants_detail"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        previous_results = tracker.get_slot("results")
        question = tracker.latest_message["text"]
        answer = chatGPT.ask(previous_results, question)
        dispatcher.utter_message(text = answer)


class ActionRestaurantsDistance(Action):
    def name(self) -> Text:
        return "action_distance"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        previous_results = tracker.get_slot("results")
        func_calling= ask_distance(previous_results)
        reply_content = func_calling.to_dict()['function_call']['arguments']
        distance = json.load(reply_content)['distance']
        dispatcher.utter_message(text = get_distance(distance))




class ValidateAppointmentForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_appointment_form"


    def validate_place(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        if value.lower() == "unknown":
            dispatcher.utter_message("Please provide a valid place.")
            return {"place": None}
        else:
            return {"place": value}

    def validate_name(
        self,
        value: Text,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        if value.lower() == "unknown":
            dispatcher.utter_message("Please provide a valid name.")
            return {"name": None}
        else:
            return {"name": value}


class ActionBookAppointment(Action):
    def name(self) -> Text:
        return "action_book_appointment"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        place = tracker.get_slot("place")

        # Perform the booking logic here
        # You can use the collected information (name, place) to book the appointment

        # For demonstration purposes, let's just confirm the booking in the response
        dispatcher.utter_message(f"Appointment booked for {name} in {place}")

        return []


class ActionBookAppointmentMR(Action):
    def name(self) -> Text:
        return "action_book_appointmentmr"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        place = tracker.get_slot("place")
        number = tracker.get_slot("number")

        # Perform the booking logic here
        # You can use the collected information (name, place) to book the appointment

        # For demonstration purposes, let's just confirm the booking in the response
        dispatcher.utter_message(f"Appointment booked for {name} in {place}. with MRnumber")

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



from rasa_sdk import Tracker, FormValidationAction, Action
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

    
    
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
            dispatcher.utter_message(
                text=f"I don't recognize that pizza. We serve {'/'.join(ALLOWED_PIZZA_TYPES)}."
            )
            return {"pizza_type": None}
        dispatcher.utter_message(text=f"OK! You want to have a {slot_value} pizza.")
        return {"pizza_type": slot_value}


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
    
    


class ActionSayData(Action):
    def name(self) -> Text:
        return "action_say_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        place = tracker.get_slot("place")

        # Perform the booking logic here
        # You can use the collected information (name, place) to book the appointment

        # For demonstration purposes, let's just confirm the booking in the response
        dispatcher.utter_message(f"Appointment booked for {name} in {place} (using rules).")

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
