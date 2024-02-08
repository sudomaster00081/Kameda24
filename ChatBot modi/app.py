from flask import Flask, render_template, request, jsonify

department = ''
date = ''
doctor = ''
time = ''
booking = False

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    global department, date, doctor, time, booking
    
    msg = request.form["msg"]
    response = {}

    if not booking:
        if msg.lower() == 'enquiry':
            response['chat'] = "Welcome to our inquiry service. How may we assist you today?"
            response['options'] = ['Go Back']
        elif msg.lower() == 'book appointment':
            booking = True
            response['chat'] = "Great! You've chosen to book an appointment. Please select a department."
            response['options'] = ['Cardiology', 'Orthopedics', 'Dermatology', 'Go Back']
        elif msg.lower() == 'about':
            response['chat'] = "Sure, here's some information about our system:\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur et nisi vel ex malesuada congue. Vestibulum nec eros nec eros scelerisque consequat. Duis vel sem vel mauris consequat interdum vel a urna. Proin consectetur dui id posuere commodo."
            response['options'] = ['Go Back']
        else:
            response['chat'] = "Hello! How can I assist you today?"
            response['options'] = ['Enquiry', 'Book Appointment', 'About']
    else:
        if not department:
            if msg.lower() == 'go back':
                booking = False
                response['chat'] = "You've returned to the main menu. How can I assist you?"
                response['options'] = ['Enquiry', 'Book Appointment', 'About']
            elif msg.lower() in ['cardiology', 'orthopedics', 'dermatology']:
                department = msg
                response['chat'] = f"Great choice! You've selected the {department} department. Now, choose a date."
                response['options'] = ['Today', 'Tomorrow', 'Next Week', 'Go Back']
            else:
                response['chat'] = "Please choose a valid department."
                response['options'] = ['Cardiology', 'Orthopedics', 'Dermatology', 'Go Back']
        elif not date:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the department selection. Please choose a department."
                response['options'] = ['Cardiology', 'Orthopedics', 'Dermatology', 'Go Back']
                # Reset variables for the current booking
                department = ''
                doctor = ''
                time = ''
            elif msg.lower() in ['today', 'tomorrow', 'next week']:
                date = msg
                response['chat'] = f"Perfect! You've selected {date}. Now, choose a doctor."
                response['options'] = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Go Back']
            else:
                response['chat'] = "Please choose a valid date."
                response['options'] = ['Today', 'Tomorrow', 'Next Week', 'Go Back']
        elif not doctor:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the date selection. Please choose a date."
                response['options'] = ['Today', 'Tomorrow', 'Next Week', 'Go Back']
                # Reset variables for the current booking
                doctor = ''
                time = ''
            elif msg.lower() in ['dr. smith', 'dr. johnson', 'dr. williams']:
                doctor = msg
                response['chat'] = f"Excellent choice! You've selected {doctor}. Now, choose a time."
                response['options'] = ['Morning', 'Afternoon', 'Evening', 'Go Back']
            else:
                response['chat'] = "Please choose a valid doctor."
                response['options'] = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Go Back']
        elif not time:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the doctor selection. Please choose a doctor."
                response['options'] = ['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Go Back']
                # Reset variables for the current booking
                time = ''
            elif msg.lower() in ['morning', 'afternoon', 'evening']:
                time = msg
                response['chat'] = f"Fantastic! You've selected {time}. Please confirm your appointment:\n\nDepartment: {department}\nDate: {date}\nDoctor: {doctor}\nTime: {time}\n\n"
                response['options'] = ['Confirm', 'Reset', 'Go Back']
            else:
                response['chat'] = "Please choose a valid time."
                response['options'] = ['Morning', 'Afternoon', 'Evening', 'Go Back']
        elif msg.lower() == 'confirm':
            response['chat'] = f"Appointment confirmed!\n\nDepartment: {department}\nDate: {date}\nDoctor: {doctor}\nTime: {time}\n\nThank you!"
            # Reset variables for the next booking
            department = ''
            date = ''
            doctor = ''
            time = ''
            booking = False
        elif msg.lower() == 'reset':
            response['chat'] = "Appointment selection reset. Please choose an department."
            response['options'] = ['Cardiology', 'Orthopedics', 'Dermatology', 'Go Back']
            # Reset variables for the current booking
            department = ''
            date = ''
            doctor = ''
            time = ''
        elif msg.lower() == 'go back':
            response['chat'] = "You've returned to the main menu. How can I assist you?"
            response['options'] = ['Enquiry', 'Book Appointment', 'About']
            # Reset variables for the current booking
            date = ''
            doctor = ''
            time = ''
        else:
            response['chat'] = "You've already completed the selection. If you want to start over, please refresh the page."

    return jsonify(response)

if __name__ == '__main__':
    app.run()
