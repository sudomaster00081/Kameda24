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
            response['chat'] = "You selected enquiry. How can we assist you?"
            response['options'] = ['Go Back']
        elif msg.lower() == 'book appointment':
            booking = True
            response['chat'] = "You selected to book an appointment. Please choose a department."
            response['options'] = ['Department A', 'Department B', 'Department C', 'Go Back']
        elif msg.lower() == 'about':
            response['chat'] = "You selected about. Here is some information about our system:\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur et nisi vel ex malesuada congue. Vestibulum nec eros nec eros scelerisque consequat. Duis vel sem vel mauris consequat interdum vel a urna. Proin consectetur dui id posuere commodo."
            response['options'] = ['Go Back']
        else:
            response['chat'] = "Hey, how can I help you?"
            response['options'] = ['Enquiry', 'Book Appointment', 'About']
    else:
        if not department:
            if msg.lower() == 'go back':
                booking = False
                response['chat'] = "You've returned to the main menu. How can I assist you?"
                response['options'] = ['Enquiry', 'Book Appointment', 'About']
            elif msg.startswith('Department'):
                department = msg
                response['chat'] = f"You selected {department}. Now, choose a date."
                response['options'] = ['Date 1', 'Date 2', 'Date 3', 'Go Back']
            else:
                response['chat'] = "Please choose a valid department."
                response['options'] = ['Department A', 'Department B', 'Department C', 'Go Back']
        elif not date:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the department selection. Please choose a department."
                response['options'] = ['Department A', 'Department B', 'Department C', 'Go Back']
                # Reset variables for the current booking
                department = ''
                doctor = ''
                time = ''
            elif msg.startswith('Date'):
                date = msg
                response['chat'] = f"You selected {date}. Now, choose a doctor."
                response['options'] = ['Doctor A', 'Doctor B', 'Doctor C', 'Go Back']
            else:
                response['chat'] = "Please choose a valid date."
                response['options'] = ['Date 1', 'Date 2', 'Date 3', 'Go Back']
        elif not doctor:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the date selection. Please choose a date."
                response['options'] = ['Date 1', 'Date 2', 'Date 3', 'Go Back']
                # Reset variables for the current booking
                doctor = ''
                time = ''
            elif msg.startswith('Doctor'):
                doctor = msg
                response['chat'] = f"You selected {doctor}. Now, choose a time."
                response['options'] = ['Time 1', 'Time 2', 'Time 3', 'Go Back']
            else:
                response['chat'] = "Please choose a valid doctor."
                response['options'] = ['Doctor A', 'Doctor B', 'Doctor C', 'Go Back']
        elif not time:
            if msg.lower() == 'go back':
                response['chat'] = "You've returned to the doctor selection. Please choose a doctor."
                response['options'] = ['Doctor A', 'Doctor B', 'Doctor C', 'Go Back']
                # Reset variables for the current booking
                time = ''
            elif msg.startswith('Time'):
                time = msg
                response['chat'] = f"You selected {time}. Please confirm your appointment: \n\nDepartment: {department}\nDate: {date}\nDoctor: {doctor}\nTime: {time}\n\n"
                response['options'] = ['Confirm', 'Reset', 'Go Back']
            else:
                response['chat'] = "Please choose a valid time."
                response['options'] = ['Time 1', 'Time 2', 'Time 3', 'Go Back']
        elif msg.lower() == 'confirm':
            response['chat'] = f"Appointment confirmed!\n\nDepartment: {department}\nDate: {date}\nDoctor: {doctor}\nTime: {time}\n\nThank you!"
            # Reset variables for the next booking
            department = ''
            date = ''
            doctor = ''
            time = ''
            booking = False
        elif msg.lower() == 'reset':
            response['chat'] = "Appointment selection reset. Please choose a department."
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
