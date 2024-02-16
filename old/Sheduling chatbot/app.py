from flask import Flask, render_template, request

app = Flask(__name__)

# Sample data for departments, doctors, dates, and times (replace with your actual data)
departments_data = {'Cardiology': ['Dr. Smith', 'Dr. Johnson'],
                    'Orthopedics': ['Dr. Davis', 'Dr. White'],
                    'Dermatology': ['Dr. Brown', 'Dr. Miller']}

dates_data = ['2024-02-10', '2024-02-11', '2024-02-12']
times_data = ['09:00 AM', '02:00 PM', '04:30 PM']

# Conversation state dictionary to store user selections
user_selections = {}

@app.route('/')
def index():
    # Display greetings and hospital info
    return render_template('greetings.html')

@app.route('/book_appointment')
def book_appointment():
    # Display department selection buttons
    return render_template('departments.html', departments=departments_data.keys())

@app.route('/select_date', methods=['GET', 'POST'])
def select_date():
    if request.method == 'POST':
        # Handle form submission
        user_selections['department'] = request.form['department']
    # Display date selection buttons
    return render_template('dates.html', dates=dates_data)

@app.route('/select_doctor', methods=['GET', 'POST'])
def select_doctor():
    if request.method == 'POST':
        # Handle form submission
        user_selections['date'] = request.form['date']
    # Display doctor selection buttons based on the department
    selected_department = user_selections.get('department')
    doctors = departments_data.get(selected_department, [])
    return render_template('doctors.html', doctors=doctors)

@app.route('/select_time', methods=['GET', 'POST'])
def select_time():
    if request.method == 'POST':
        # Handle form submission
        user_selections['doctor'] = request.form['doctor']
    # Display time selection buttons
    return render_template('times.html', times=times_data)

@app.route('/confirmation', methods=['GET', 'POST'])
def confirmation():
    if request.method == 'POST':
        # Handle form submission
        user_selections['time'] = request.form['time']
    # Display confirmation page with user selections
    return render_template('confirmation.html', selections=user_selections)

if __name__ == '__main__':
    app.run(debug=True)
