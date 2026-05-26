import sched
# pytest testing script. all functions prefixed with 'test_' will be auto-discovered and run by pytest
from src import add
from src.VitalSignsMonitor import VitalSignsMonitor
from src.MedicationTracker import MedicationTracker
from src.AppointmentSchdeuler import AppointmentScheduler
from src.AppointmentScheduler2 import AppointmentScheduler2


def test_add_function():
    """Test the add function."""
    assert add(2, 3) == 5


def test_record_and_get_latest():
    """Test recording vital signs and retrieving the latest value."""
    monitor = VitalSignsMonitor("patient_123")
    monitor.record("heart_rate", 70.0, 1)
    monitor.record("heart_rate", 75.0, 2)
    assert monitor.get_latest("heart_rate") == 75.0


def test_get_average():
    """Test calculating average of vital signs."""
    monitor = VitalSignsMonitor("patient_123")
    monitor.record("heart_rate", 70.0, 1)
    monitor.record("heart_rate", 75.0, 2)
    assert monitor.get_average("heart_rate", 1, 2) == 72.5


def test_alert_triggers():
    """Test alert triggering when values are out of range."""
    monitor = VitalSignsMonitor("patient_123")
    monitor.record("heart_rate", 70.0, 1)
    monitor.record("heart_rate", 75.0, 2)
    monitor.set_alert("heart_rate", 60.0, 100.0)

    assert monitor.record("heart_rate", 55.0, 3) is True  # alert triggered (below min)
    assert monitor.record("heart_rate", 80.0, 4) is False  # within range
    assert monitor.record("heart_rate", 105.0, 5) is True  # alert triggered (above max)


def test_nonexistent_vital_sign():
    """Test behavior with non-existent vital signs."""
    monitor = VitalSignsMonitor("patient_123")
    assert monitor.get_latest("blood_pressure") is None
    assert monitor.get_average("blood_pressure", 0, 10) is None


def test_record_new_vital_sign():
    """Test recording a new type of vital sign."""
    monitor = VitalSignsMonitor("patient_123")
    monitor.record("blood_pressure", 120.0, 1)
    assert monitor.get_latest("blood_pressure") == 120.0
    assert monitor.get_average("blood_pressure", 0, 2) == 120.0


def test_add_medication():
    """Test adding a new medication."""
    tracker = MedicationTracker("kitty_456")
    assert tracker.add_medication("Aspirin", 100, 8) is True


def test_add_duplicate_medication():
    """Test that adding a duplicate medication fails."""
    tracker = MedicationTracker("kitty_456")
    tracker.add_medication("Aspirin", 100, 8)
    assert tracker.add_medication("Aspirin", 100, 12) is False


def test_add_multiple_medications():
    """Test adding multiple different medications."""
    tracker = MedicationTracker("kitty_456")
    assert tracker.add_medication("Aspirin", 100, 8) is True
    assert tracker.add_medication("Ibuprofen", 200, 6) is True


def test_get_medications():
    """Test retrieving all medications."""
    tracker = MedicationTracker("kitty_456")
    tracker.add_medication("Aspirin", 100, 8)
    tracker.add_medication("Ibuprofen", 200, 6)

    meds = tracker.get_medications()
    assert len(meds) == 2
    assert any(
        med["name"] == "Aspirin"
        and med["dosage_mg"] == 100
        and med["frequency_hours"] == 8
        for med in meds
    )
    assert any(
        med["name"] == "Ibuprofen"
        and med["dosage_mg"] == 200
        and med["frequency_hours"] == 6
        for med in meds
    )

def test_record_dose():
    """Test recording a dose for a medication."""
    tracker = MedicationTracker("kitty_456")
    tracker.add_medication("Aspirin", 100, 8)
    assert tracker.record_dose("Aspirin", 1) is True

def test_record_dose_nonexistent_medication():
    """Test recording a dose for a non-existent medication."""
    tracker = MedicationTracker("kitty_456")
    assert tracker.record_dose("NonExistentMed", 1700000) is False

def test_last_dose_recorded():
    """Test that the last dose recorded is correct."""
    tracker = MedicationTracker("kitty_456")
    tracker.add_medication("Aspirin", 100, 8)
    tracker.record_dose("Aspirin", 1700000)
    tracker.record_dose("Aspirin", 1703600)
    assert tracker.get_last_dose_time("Aspirin") == 1703600

def test_last_dose_no_doses():
    """Test that None is returned if no doses have been recorded."""
    tracker = MedicationTracker("kitty_456")
    tracker.add_medication("Aspirin", 100, 8)
    assert tracker.get_last_dose_time("Aspirin") is None

def test_last_dose_nonexistent_medication():
    """Test that None is returned for last dose time of a non-existent medication."""
    tracker = MedicationTracker("kitty_456")
    assert tracker.get_last_dose_time("NonExistentMed") is None

def test_is_dose_due():
    """Test checking if a dose is due based on last dose time and frequency."""
    tracker = MedicationTracker("kitty_456")
    tracker.add_medication("Aspirin", 100, 8)  # every 8 hours
    assert tracker.is_dose_due("Aspirin", 1700000) is True  # never taken, so due

    tracker.record_dose("Aspirin", 1700000)  # taken at time 1700000
    assert tracker.is_dose_due("Aspirin", 1700000 + 7*3600) is False  # not yet due
    assert tracker.is_dose_due("Aspirin", 1700000 + 8*3600) is True  # now due

def test_is_dose_due_nonexistent_medication():
    """Test checking if a dose is due for a non-existent medication."""
    tracker = MedicationTracker("kitty_456")
    assert tracker.is_dose_due("NonExistentMed", 1700000) is False

def test_get_overdue_medications():
    """Test retrieving a list of overdue medications."""
    tracker = MedicationTracker("kitty_456")
    tracker.add_medication("Aspirin", 100, 8)  # every 8 hours
    tracker.add_medication("Ibuprofen", 200, 6)  # every 6 hours

    tracker.record_dose("Aspirin", 1700000)  # taken at time 1700000
    tracker.record_dose("Ibuprofen", 1700000)  # taken at time 1700000

    overdue = tracker.get_overdue_medications(1700000 + 5*3600)  # check at time 1700000 + 5 hours
    assert len(overdue) == 0  # No medications should be overdue

    overdue = tracker.get_overdue_medications(1700000 + 9*3600)  # check at time 1700000 + 9 hours
    assert "Ibuprofen" in overdue  # Ibuprofen should be overdue

    overdue = tracker.get_overdue_medications(1700000 + 11*3600)  # check at time 1700000 + 11 hours
    assert "Aspirin" in overdue  # Aspirin should be overdue
    assert "Ibuprofen" in overdue  # Ibuprofen should still be overdue


# testing AppointmentScheduler
def test_add_time_slot():
    """Test adding appointment time slots."""
    scheduler = AppointmentScheduler("Health Clinic")
    assert scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith') is True
    assert scheduler.add_time_slot('2025-12-10', '09:15', 30, 'Dr. Smith') is False  # Overlaps
    assert scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith') is True
    assert scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Jones') is True  # Different provider

def test_appointment_booking():
    """Test booking appointments."""
    scheduler = AppointmentScheduler("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')

    confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    assert confirmation == "APPT-2025-12-10-09:00-Dr. Smith"
    # try again and catch value error
    try:
        confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'Jane Doe', 'Dr. Smith')
    except ValueError:
        pass

def test_nonexistent_time_slot_booking():
    """Test booking an appointment for a non-existent time slot."""
    scheduler = AppointmentScheduler("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')

    try:
        confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    except ValueError:
        pass

def test_cancel_appointment():
    """Test cancelling an appointment."""
    scheduler = AppointmentScheduler("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')

    confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    assert confirmation == "APPT-2025-12-10-09:00-Dr. Smith"

    assert scheduler.cancel_appointment(confirmation) is True
    # try cancelling again
    assert scheduler.cancel_appointment(confirmation) is False
    confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'Jane Doe', 'Dr. Smith')
    assert confirmation == "APPT-2025-12-10-09:00-Dr. Smith"

def test_cancel_nonexistent_appointment():
    """Test cancelling a non-existent appointment."""
    scheduler = AppointmentScheduler("Health Clinic")
    assert scheduler.cancel_appointment("NONEXISTENT-APPT-ID") is False

def test_get_available_slots():
    """Test retrieving available time slots for a provider on a given date."""
    scheduler = AppointmentScheduler("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '11:00', 30, 'Dr. Smith')

    confirmation = scheduler.book_appointment('2025-12-10', '10:00', 'John Doe', 'Dr. Smith')

    available_slots = scheduler.get_available_slots('2025-12-10', 'Dr. Smith')
    assert len(available_slots) == 2

def test_get_available_slots_no_slots():
    """Test retrieving available time slots when none exist."""
    scheduler = AppointmentScheduler("Health Clinic")
    available_slots = scheduler.get_available_slots('2025-12-10', 'Dr. Smith')
    assert len(available_slots) == 0

def test_get_available_slots_all_booked():
    """Test retrieving available time slots when all are booked."""
    scheduler = AppointmentScheduler("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')

    scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    scheduler.book_appointment('2025-12-10', '10:00', 'Jane Doe', 'Dr. Smith')

    available_slots = scheduler.get_available_slots('2025-12-10', 'Dr. Smith')
    assert len(available_slots) == 0

def test_get_available_slots_provider_agnostic():
    """Test retrieving available time slots for different providers."""
    scheduler = AppointmentScheduler("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Jones')

    available_slots_smith = scheduler.get_available_slots('2025-12-10', 'Dr. Smith')
    available_slots_jones = scheduler.get_available_slots('2025-12-10', 'Dr. Jones')
    available_slots_who = scheduler.get_available_slots('2025-12-10', 'Dr. Who')
    available_slots_all = scheduler.get_available_slots('2025-12-10')

    assert len(available_slots_smith) == 1
    assert len(available_slots_jones) == 1
    assert len(available_slots_who) == 0
    assert len(available_slots_all) == 2

def test_get_patient_appointments():
    """Test retrieving all appointments for a specific patient."""
    scheduler = AppointmentScheduler("Health Clinic")
    assert scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    assert scheduler.add_time_slot('2025-12-11', '09:30', 30, 'Dr. Jones')
    assert scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')
    assert scheduler.add_time_slot('2025-12-12', '11:00', 30, 'Dr. Who')

    conf1 = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    conf2 = scheduler.book_appointment('2025-12-11', '09:30', 'John Doe', 'Dr. Jones')
    conf3 = scheduler.book_appointment('2025-12-10', '10:00', 'Jane Doe', 'Dr. Smith')

    john_appointments = scheduler.get_patient_appointments('John Doe')
    jane_appointments = scheduler.get_patient_appointments('Jane Doe')
    assert len(john_appointments) == 2
    assert len(jane_appointments) == 1

# testing AppointmentScheduler2

def test_time_from_string2():
    scheduler = AppointmentScheduler2("Health Clinic")
    assert scheduler._time_from_string("09:00") == 9*60
    assert scheduler._time_from_string("12:30") == 12*60 + 30
    assert scheduler._time_from_string("00:00") == 0*60
    assert scheduler._time_from_string("23:59") == 23*60 + 59

def test_time_to_string2():
    scheduler = AppointmentScheduler2("Health Clinic")
    assert scheduler._time_to_string(9*60) == "09:00"
    assert scheduler._time_to_string(12*60 + 30) == "12:30"
    assert scheduler._time_to_string(0*60) == "00:00"
    assert scheduler._time_to_string(23*60 + 59) == "23:59"

def test_get_initials2():
    scheduler = AppointmentScheduler2("Health Clinic")
    assert scheduler._get_initials("Dr. John Smith") == "DJS"
    assert scheduler._get_initials("Nurse Mary Jane") == "NMJ"
    assert scheduler._get_initials("Dr. Who") == "DW"
    assert scheduler._get_initials("SingleName") == "S"

def test_check_overlap2():
    scheduler = AppointmentScheduler2("Health Clinic")
    assert scheduler._check_overlap(0, 30, 15, 45) is True # b overlaps a
    assert scheduler._check_overlap(15, 45, 0, 30) is True # a overlaps b
    assert scheduler._check_overlap(0, 30, 30, 60) is False # touching but not overlapping
    assert scheduler._check_overlap(0, 30, 60, 90) is False # completely separate
    assert scheduler._check_overlap(60, 90, 0, 30) is False # completely separate
    assert scheduler._check_overlap(0, 60, 15, 45) is True # b inside a
    assert scheduler._check_overlap(15, 45, 0, 60) is True # a inside b

def test_add_time_slot2():
    scheduler = AppointmentScheduler2("Health Clinic")
    assert scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith') is True
    assert scheduler.add_time_slot('2025-12-10', '09:15', 30, 'Dr. Smith') is False  # Overlaps
    assert scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith') is True
    assert scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Jones') is True  # Different provider

def test_book_appointment2():
    scheduler = AppointmentScheduler2("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '11:00', 30, 'Dr. Jones')
    scheduler.add_time_slot('2025-12-11', '09:00', 30, 'Dr. Smith')

    confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    assert confirmation == "APPT-2025-12-10-0900-DS"
    # try again and catch value error
    try:
        confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'Jane Doe', 'Dr. Smith')
    except ValueError:
        pass
    confirmation2 = scheduler.book_appointment('2025-12-10', '11:00', 'Jane Doe', 'Dr. Jones')
    assert confirmation2 == "APPT-2025-12-10-1100-DJ"

def test_cancel_appointment2():
    scheduler = AppointmentScheduler2("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')

    confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    assert confirmation == "APPT-2025-12-10-0900-DS"

    assert scheduler.cancel_appointment(confirmation) is True
    assert scheduler.cancel_appointment(confirmation) is False
    confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'Jane Doe', 'Dr. Smith')
    assert confirmation == "APPT-2025-12-10-0900-DS"

    assert scheduler.cancel_appointment(confirmation) is True
    assert scheduler.cancel_appointment(confirmation) is False

    assert scheduler.cancel_appointment("NONEXISTENT-APPT-ID") is False
    
def test_get_available_slots2():
    scheduler = AppointmentScheduler2("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '11:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Jones')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Jones')

    assert len(scheduler.get_available_slots('2025-12-10', 'Dr. Smith')) == 3
    assert len(scheduler.get_available_slots('2025-12-10', 'Dr. Jones')) == 2

    confirmation = scheduler.book_appointment('2025-12-10', '10:00', 'John Doe', 'Dr. Smith')
    assert len(scheduler.get_available_slots('2025-12-10', 'Dr. Smith')) == 2
    scheduler.cancel_appointment(confirmation)
    assert len(scheduler.get_available_slots('2025-12-10', 'Dr. Smith')) == 3

def test_get_patient_appointments2():
    scheduler = AppointmentScheduler2("Health Clinic")
    assert scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    assert scheduler.add_time_slot('2025-12-11', '09:30', 30, 'Dr. Jones')
    assert scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')
    assert scheduler.add_time_slot('2025-12-12', '11:00', 30, 'Dr. Who')

    confirmation0 = scheduler.book_appointment('2025-12-10', '10:00', 'Jane Doe', 'Dr. Smith')
    jane_appointments = scheduler.get_patient_appointments('Jane Doe')
    assert len(jane_appointments) == 1

    john_appointments = scheduler.get_patient_appointments('John Doe')
    assert len(john_appointments) == 0

    confirmation1 = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    john_appointments = scheduler.get_patient_appointments('John Doe')
    assert len(john_appointments) == 1

    confirmation2 = scheduler.book_appointment('2025-12-11', '09:30', 'John Doe', 'Dr. Jones')
    john_appointments = scheduler.get_patient_appointments('John Doe')
    assert len(john_appointments) == 2

    assert scheduler.cancel_appointment(confirmation1) is True
    john_appointments = scheduler.get_patient_appointments('John Doe')
    assert len(john_appointments) == 1

    assert scheduler.cancel_appointment(confirmation2) is True
    john_appointments = scheduler.get_patient_appointments('John Doe')
    assert len(john_appointments) == 0

    assert scheduler.cancel_appointment(confirmation2) is False
    john_appointments = scheduler.get_patient_appointments('John Doe')
    assert len(john_appointments) == 0

    assert scheduler.cancel_appointment(confirmation0) is True
    jane_appointments = scheduler.get_patient_appointments('Jane Doe')
    assert len(jane_appointments) == 0

def test_reschedule_appointment2():
    scheduler = AppointmentScheduler2("Health Clinic")
    scheduler.add_time_slot('2025-12-10', '09:00', 30, 'Dr. Smith')
    scheduler.add_time_slot('2025-12-10', '10:00', 30, 'Dr. Smith')

    confirmation = scheduler.book_appointment('2025-12-10', '09:00', 'John Doe', 'Dr. Smith')
    assert confirmation == "APPT-2025-12-10-0900-DS"

    assert scheduler.reschedule_appointment(confirmation, '2025-12-10', '10:00') is True
    assert scheduler.reschedule_appointment(confirmation, '2025-12-10', '10:00') is False  # slot already booked