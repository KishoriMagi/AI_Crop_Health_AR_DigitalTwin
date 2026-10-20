import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from database import init_db, get_db_connection
from dropbox_service import upload_test_request, fetch_and_process_results
import uuid

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret')

# Initialize DB on startup
init_db()

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    conn = get_db_connection()
    patients_count = conn.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
    tests = conn.execute('SELECT status, COUNT(*) FROM test_requests GROUP BY status').fetchall()
    
    stats = {
        'total_patients': patients_count,
        'pending': 0,
        'sent': 0,
        'completed': 0
    }
    
    for row in tests:
        if row['status'] == 'Pending':
            stats['pending'] = row[1]
        elif row['status'] == 'Sent to Laboratory':
            stats['sent'] = row[1]
        elif row['status'] == 'Result Available':
            stats['completed'] = row[1]
            
    recent_tests = conn.execute('''
        SELECT t.*, p.name as patient_name 
        FROM test_requests t 
        JOIN patients p ON t.patient_id = p.patient_id 
        ORDER BY id DESC LIMIT 5
    ''').fetchall()
    
    conn.close()
    return render_template('dashboard.html', stats=stats, recent_tests=recent_tests)

@app.route('/patients')
def patients_page():
    conn = get_db_connection()
    patients = conn.execute('SELECT * FROM patients ORDER BY id DESC').fetchall()
    conn.close()
    return render_template('patients.html', patients=patients)

@app.route('/api/patients', methods=['GET', 'POST'])
def api_patients():
    conn = get_db_connection()
    if request.method == 'POST':
        data = request.json
        patient_id = f"P{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            conn.execute('''
                INSERT INTO patients (patient_id, name, age, gender, phone, address, blood_group)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (patient_id, data['name'], data['age'], data['gender'], data['phone'], data['address'], data['blood_group']))
            conn.commit()
            return jsonify({'success': True, 'patient_id': patient_id}), 201
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        finally:
            conn.close()
    else:
        patients = conn.execute('SELECT * FROM patients').fetchall()
        conn.close()
        return jsonify([dict(p) for p in patients])

@app.route('/test_requests')
def test_requests_page():
    conn = get_db_connection()
    patients = conn.execute('SELECT patient_id, name FROM patients').fetchall()
    tests = conn.execute('''
        SELECT t.*, p.name as patient_name 
        FROM test_requests t 
        JOIN patients p ON t.patient_id = p.patient_id 
        ORDER BY t.id DESC
    ''').fetchall()
    conn.close()
    return render_template('test_requests.html', patients=patients, tests=tests)

@app.route('/api/test-requests', methods=['POST'])
def api_create_test_request():
    data = request.json
    test_id = f"T{datetime.now().strftime('%Y%m%d%H%M%S')}"
    conn = get_db_connection()
    
    try:
        conn.execute('''
            INSERT INTO test_requests (test_id, patient_id, doctor, test_type, sample_type, priority, date, clinical_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (test_id, data['patient_id'], data['doctor'], data['test_type'], data['sample_type'], data['priority'], datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data.get('clinical_notes', '')))
        conn.commit()
        
        # Prepare data for Dropbox CSV
        csv_data = {
            'TestID': test_id,
            'PatientID': data['patient_id'],
            'TestType': data['test_type'],
            'SampleType': data['sample_type'],
            'Priority': data['priority'],
            'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Notes': data.get('clinical_notes', '')
        }
        
        # Upload to Dropbox
        upload_path = upload_test_request(test_id, csv_data)
        
        if upload_path:
            conn.execute("UPDATE test_requests SET status='Sent to Laboratory' WHERE test_id=?", (test_id,))
            conn.commit()
            status = 'Sent to Laboratory'
        else:
            status = 'Pending' # Failed to upload
            
        return jsonify({'success': True, 'test_id': test_id, 'status': status}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    finally:
        conn.close()

@app.route('/api/dropbox/sync-results', methods=['POST'])
def api_sync_results():
    results = fetch_and_process_results()
    conn = get_db_connection()
    
    synced_count = 0
    for res in results:
        data = res['data']
        test_id = data.get('TestID')
        if test_id:
            # Store result as JSON string in DB
            result_json = json.dumps(data)
            conn.execute('''
                UPDATE test_requests 
                SET status='Result Available', result_data=? 
                WHERE test_id=?
            ''', (result_json, test_id))
            synced_count += 1
            
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'synced_count': synced_count})

@app.route('/api/results/<test_id>')
def api_get_result(test_id):
    conn = get_db_connection()
    test = conn.execute('SELECT * FROM test_requests WHERE test_id=?', (test_id,)).fetchone()
    conn.close()
    
    if test and test['result_data']:
        return jsonify(json.loads(test['result_data']))
    return jsonify({'error': 'Result not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
