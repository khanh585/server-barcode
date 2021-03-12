import pyrebase

def init_storage():
    config={    "apiKey": "AIzaSyDMLcbg9Mui1sLWFMdNGea-hhYKFVz9rdI",
                "authDomain": "capstone-96378.firebaseapp.com",
                "projectId": "capstone-96378",
                "storageBucket": "capstone-96378.appspot.com",
                "databaseURL": "capstone-96378.appspot.com",
                "messagingSenderId": "544847549773",
                "appId": "1:544847549773:web:1b199598f8761a21761304",
                "measurementId": "G-7P4XGE0H3M",
                "type": "service_account",
                "project_id": r"capstone-96378",
                "private_key_id": r"addbc78d5ac83b464fe7aa0e47f2a9d2e8f375d7",
                "private_key": r"-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQCvJQZaerM62BDt\ne2oGXDcaJrjv9WuD9jUj4+qRX8ybJEixGNQXIzekxD2MJ67ZlwRTdBHGXgdGRocY\n/e6pZHXgEjZpZrKhYCCjsMY8U7rQSjLRE1J/HOxK6Lc+k+kJA1QS6BI6cQYTcNeP\nvUEjwiyro6q22pM75LDl0ujT3TedoKd2/8XwSwRIH51UkKYnBv9yIa1GkCKYSYjT\nZwA7IWxaKi744V13VJ+r57f4KVXB9ARiQWNpsam+y1mHsaWYLqmmeYCLxgl9Mi1O\ntdBsw62Libmg90hht/r0k4qVdZcuBzxEN8mPRwtZOgll+cZH+YIHq2Lqo9Umn87X\nFp/DSaDPAgMBAAECggEAEuZpGfDIEDoKaEOF91C9MNPBPO0cIJM8PHbBtIAE2YW1\nWHMS9A+SAh7xJgOO6/PO0jNMyxNmL+W4JMYbmpOjir4RuWxigLqZsGsI7AVRvAdo\na2/ChRgPFwL86e0BMT6hicwZp3fUjYWC+tkT7fLBakuQ8QbY65Yr/BO3RKGdzk2/\nANkyyL1x9YhurG/GPzlaXYku252sfW9nTO0Cl3W0lnVGx0ARniHmbALiUOfzewa8\nMrJz9fSSX3Tu8aEfIPamy9svT5tPOzxYPU9ZIBCHRcY3gAvQ91aDLHy+2+XAth6C\nAgsysr2VEc7ALExPoqLmkqeaiwbk2KgdTCS38eOGTQKBgQDVgxlSl0jI40XteTEd\n7l1C5uJkIAb27Grl4guj+oawAr4xOKcftx2oIIxPNxLskcYPvQL6OdWiUtwRtta1\n7+DTE9mvR98sjrhe1yp518bn5A1vEOjf2oVeURZskWeSbSZyFHa4L/OzVIMimqRi\nDQGFITPDozoXx/MfK6AjTohs4wKBgQDR/2E671mNlTWGxtOWzy2AOshlfrg5dwlB\n5sgirJKyrsXqzNYH+iMlfsXsOMeIcjo7FWw2y9k97dnlocJJso5dPGgjdHIAut9Z\neYUbwch+40X4zjWTCZECq17tbTTkO+POyCdDzHosXgi6djOv9X83ihbI4+BOPUh5\nGP36oI/MJQKBgQCDQB7t7X/ZthSDge+WpbQA72uV/zYoznZ8MHPVbqkjQ9xk552c\n2nVJqBB4lbN1Z7soa5mL9seEDImp/gqJNxeuydIUdQsVGfrumjhLREDKxh8X+n4d\nNfisXQ562PuhQKXUlQ3R9fBKehZOzF9mwvX/P8TRa+LHQawLqka7zYgB3wKBgQCQ\nbCJiddoRFV0KpBNA1JtmZQcNRmlMnCExwZZCwozhdnGHWfiY497oGpfkLeiMXoLN\nv5380ZVGkh4ZX3ZPWQ2v8attNthwyBRzZK+2tz88hPHKe4c86IdiXpmakFCf5tB1\nlhHq/uXHPaSjGEfT6/LiL4YcAxxwspT1M8r4YxSCQQKBgQCOkb16/kfmoBqcDGim\n53JK6lHSU/hWeYLfJ9h3QzAWsKKKELOLmYB1q+mY/LJ9z5zI0kP68X2pLbkWf7Ew\nbT+Mlj7j9Mhm9c4Jw/EQgS1gz3kuowBpTYn9jSjBDVnsSQXKzNTyG5Z6dxRTueYR\nAxESf7vQZo8335sOj2qqvqvg0g==\n-----END PRIVATE KEY-----\n",
                "client_email": r"firebase-adminsdk-t4ege@capstone-96378.iam.gserviceaccount.com",
                "client_id": r"108888086635110828498",
                "auth_uri": r"https://accounts.google.com/o/oauth2/auth",
                "token_uri": r"https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": r"https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": r"https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-t4ege%40capstone-96378.iam.gserviceaccount.com"
            }
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    return storage