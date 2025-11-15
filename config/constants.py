TRANSACTION_TYPES = ['Income', 'Expense', 'To Receive', 'To Pay']

CATEGORIES = {
    'Expense': {
        'Food': [
            'Groceries', 'Dining Out', 'Snacks', 'Vegetables',
            'Fruits', 'Beverages', 'Daily Essentials', 'Supermarket'
        ],
        'Transportation': [
            'Fuel', 'Public Transit', 'Taxi', 'Bike Maintenance', 'Car Maintenance'
        ],
        'Housing': [
            'Rent', 'Utilities', 'Maintenance', 'Repairs', 'Internet', 'Water Bill'
        ],
        'Entertainment': [
            'Movies', 'Games', 'Events', 'Streaming', 'Concerts', 'Subscriptions'
        ],
        'Shopping': [
            'Clothes', 'Electronics', 'Home Items', 'Accessories', 'Decor', 'Appliances'
        ],
        'Healthcare': [
            'Medical', 'Pharmacy', 'Insurance', 'Doctor Visit', 'Hospital', 'Medicines'
        ],
        'Gift': [
            'Birthday', 'Wedding', 'Holiday', 'Festive', 'Donation', 'Other'
        ],
        'Other': [
            'Miscellaneous', 'Unspecified', 'Unknown'
        ]
    },
    'Income': {
        'Salary': ['Regular', 'Bonus', 'Overtime'],
        'Investment': ['Dividends', 'Interest', 'Capital Gains'],
        'Other': ['Gifts', 'Refunds', 'Miscellaneous']
    },
    'To Receive': {
        'Pending Income': ['Salary', 'Investment', 'Other']
    },
    'To Pay': {
        'Bills': ['Utilities', 'Rent', 'Credit Card', 'Loan', 'Other']
    }
}
