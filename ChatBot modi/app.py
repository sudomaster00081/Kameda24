from flask import Flask, render_template, request, jsonify

# Dummy data for departments, doctors, and time slots
departments = ['Cardiology', 'Orthopedics', 'Dermatology']
doctors = {
    'Cardiology': ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams'],
    'Orthopedics': ['Dr. Davis', 'Dr. Wilson', 'Dr. Brown'],
    'Dermatology': ['Dr. Miller', 'Dr. Moore', 'Dr. Taylor']
}
time_slots = ['Morning', 'Afternoon', 'Evening']

# Booking details
booking = {
    'department': '',
    'date': '',
    'doctor': '',
    'time': '',
    'confirmed': False
}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    global booking

    msg = request.form["msg"]
    response = {}

    if not booking['confirmed']:
        if msg.lower() == 'enquiry':
            response['chat'] = "Welcome to our inquiry service. How may we assist you today?"
            response['options'] = ['Go Back']
        elif msg.lower() == 'book appointment':
            booking['confirmed'] = True
            response['chat'] = "Great! You've chosen to book an appointment. Please select a department."
            response['options'] = departments + ['Go Back']
        elif msg.lower() == 'about':
            response['chat'] = "Sure, here's some information about our system:\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur et nisi vel ex malesuada congue. Vestibulum nec eros nec eros scelerisque consequat. Duis vel sem vel mauris consequat interdum vel a urna. Proin consectetur dui id posuere commodo."
            response['options'] = ['Go Back']
        else:
            response['chat'] = "Hello! How can I assist you today?"
            response['options'] = ['Enquiry', 'Book Appointment', 'About']
    else:
        if not booking['department']:
            if msg.lower() == 'go back':
                booking['confirmed'] = False
                response['chat'] = "You've returned to the main menu. How can I assist you?"
                response['options'] = ['Enquiry', 'Book Appointment', 'About']
            elif msg.lower() in [dep.lower() for dep in departments]:
                booking['department'] = next(dep for dep in departments if dep.lower() == msg.lower())
                response['chat'] = f"Great choice! You've selected the {booking['department']} department. Now, choose a date."
                response['options'] = ['Today', 'Tomorrow', 'Next Week', 'Go Back']
            else:
                response['chat'] = "Please choose a valid department."
                response['options'] = departments + ['Go Back']
        elif not booking['date']:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the department selection. Please choose a department."
                response['options'] = departments + ['Go Back']
                # Reset variables for the current booking
                booking['department'] = ''
                booking['doctor'] = ''
                booking['time'] = ''
            elif msg.lower() in ['today', 'tomorrow', 'next week']:
                booking['date'] = msg.lower()
                response['chat'] = f"Perfect! You've selected {booking['date']}. Now, choose a doctor."
                response['options'] = doctors[booking['department']] + ['Go Back']
            else:
                response['chat'] = "Please choose a valid date."
                response['options'] = ['Today', 'Tomorrow', 'Next Week', 'Go Back']
        elif not booking['doctor']:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the date selection. Please choose a date."
                response['options'] = ['Today', 'Tomorrow', 'Next Week', 'Go Back']
                # Reset variables for the current booking
                booking['doctor'] = ''
                booking['time'] = ''
            elif msg.lower() in [doc.lower() for doc in doctors[booking['department']]]:
                booking['doctor'] = next(doc for doc in doctors[booking['department']] if doc.lower() == msg.lower())
                response['chat'] = f"Excellent choice! You've selected {booking['doctor']}. Now, choose a time."
                response['options'] = time_slots + ['Go Back']
            else:
                response['chat'] = "Please choose a valid doctor."
                response['options'] = doctors[booking['department']] + ['Go Back']
        elif not booking['time']:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the doctor selection. Please choose a doctor."
                response['options'] = doctors[booking['department']] + ['Go Back']
                # Reset variables for the current booking
                booking['time'] = ''
            elif msg.lower() in [slot.lower() for slot in time_slots]:
                booking['time'] = next(slot for slot in time_slots if slot.lower() == msg.lower())
                response['chat'] = f"Fantastic! You've selected {booking['time']}. Please confirm your appointment:\n\nDepartment: {booking['department']}\nDate: {booking['date'].capitalize()}\nDoctor: {booking['doctor']}\nTime: {booking['time']}\n\n"
                response['options'] = ['Confirm', 'Reset', 'Go Back']
            else:
                response['chat'] = "Please choose a valid time."
                response['options'] = time_slots + ['Go Back']
        elif msg.lower() == 'confirm':
            response['chat'] = f"Appointment confirmed!\n\nDepartment: {booking['department']}\nDate: {booking['date'].capitalize()}\nDoctor: {booking['doctor']}\nTime: {booking['time']}\n\nThank you!"
            # Reset variables for the next booking
            booking['department'] = ''
            booking['date'] = ''
            booking['doctor'] = ''
            booking['time'] = ''
            booking['confirmed'] = False
        elif msg.lower() == 'reset':
            response['chat'] = "Appointment selection reset. Please choose a department."
            response['options'] = departments + ['Go Back']
            # Reset variables for the current booking
            booking['department'] = ''
            booking['date'] = ''
            booking['doctor'] = ''
            booking['time'] = ''
        elif msg.lower() == 'go back':
            response['chat'] = "You've returned to the main menu. How can I assist you?"
            response['options'] = ['Enquiry', 'Book Appointment', 'About']
            # Reset variables for the current booking
            booking['date'] = ''
            booking['doctor'] = ''
            booking['time'] = ''
        else:
            response['chat'] = "You've already completed the selection. If you want to start over, please refresh the page."

    return jsonify(response)

if __name__ == '__main__':
    app.run()



# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


# tokenizer = AutoTokenizer.from_pretrained("model path")
# model = AutoModelForCausalLM.from_pretrained("model path")



