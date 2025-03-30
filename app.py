import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import requests
from bs4 import BeautifulSoup
import time
import os
import shutil
import zipfile
import pickle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float


# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sales Data Analysis", "Data Scraper", "Web File Organizer","Database Project", "Blog Application",
                                  "ML-House Price Prediction", "DL-Image Classification", "API-Integration Weather Dashboard",
                                  "Interactive Visualizations (Plotly+Dash)"])

# ---------------------------- HOME PAGE ----------------------------
if page == "Home":
    st.title("Welcome to Python Basics! 🐍")
    st.markdown("""
    ## What is Python?
    Python is a **high-level, interpreted** programming language widely used for:
    - Web Development
    - Data Science & Machine Learning
    - Automation & Scripting

    ## Key Features:
    - Readable & Simple Syntax
    - Huge Community Support
    - Libraries for Everything (NumPy, Pandas, TensorFlow, etc.)

    ## Example Python Code:
    ```python
    def greet(name):
        return f"Hello, {name}!"

    print(greet("Alice"))
    ```
    **Click on the sidebar to explore Sales Data Analysis!**
    """)

# ---------------------- SALES DATA ANALYSIS ----------------------
elif page == "Sales Data Analysis":
    st.title("Sales Data Analysis 📊")
    st.markdown("### Using California Housing Dataset as Sales Data")

    # Load dataset
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['SalesPrice'] = california.target  # Target variable

    # Display dataset
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Train Model
    X = df.drop(columns=['SalesPrice'])
    y = df['SalesPrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Model Performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Scatter Plot: Actual vs Predicted
    st.subheader("Actual vs. Predicted Sales Prices")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(y_test, y_pred, alpha=0.6, color="red")
    ax2.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="dashed", color="black")
    ax2.set_xlabel("Actual Sales Price")
    ax2.set_ylabel("Predicted Sales Price")
    ax2.set_title("Actual vs. Predicted Sales Prices")
    st.pyplot(fig2)

# ---------------------- BOOKS DATA SCRAPER ----------------------


elif page == "Data Scraper":
    st.title("Book Data Scraper 📚")
    st.markdown("### Scrape book titles & prices from [Books to Scrape](http://books.toscrape.com/)")

    # Define Scraper Function
    def scrape_books():
        url = "http://books.toscrape.com/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        books = soup.find_all("article", class_="product_pod")
        book_data = []
        for book in books:
            title = book.h3.a["title"]
            price = book.find("p", class_="price_color").text.strip()
            book_data.append([title, price])
        
        return pd.DataFrame(book_data, columns=["Title", "Price"])

    # Scrape Data on Button Click
    if st.button("Scrape Books Data"):
        with st.spinner("Scraping data... Please wait!"):
            time.sleep(2)
            book_df = scrape_books()
            st.success("Scraping Completed!")
            st.dataframe(book_df)

            # Allow CSV Download
            csv = book_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV 📥",
                data=csv,
                file_name="books_data.csv",
                mime="text/csv"
            )

# ---------------------- FILE ORGANIZER ----------------------
elif page == "File Organizer":
    st.title("File Organization Automation 📂")
    st.write("Select a folder and automatically organize files into categories!")

    # Define file categories
    FILE_TYPES = {
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx"],
        "Images": [".jpg", ".jpeg", ".png", ".gif"],
        "Videos": [".mp4", ".mov", ".avi"],
        "Music": [".mp3", ".wav", ".flac"],
        "Archives": [".zip", ".rar", ".7z"],
        "Programs": [".exe", ".msi", ".sh"]
    }

    # Function to organize files
    def organize_files(directory):
        if not os.path.exists(directory):
            return "Directory does not exist!"

        # Create folders if not exist
        for category in FILE_TYPES.keys():
            folder_path = os.path.join(directory, category)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # Move files into respective folders
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            
            if os.path.isfile(file_path):
                for category, extensions in FILE_TYPES.items():
                    if any(file.lower().endswith(ext) for ext in extensions):
                        shutil.move(file_path, os.path.join(directory, category, file))
        
        return "File organization completed!"

    # Input directory
    directory = st.text_input("Enter the directory path:")

    # Run button
    if st.button("Organize Files"):
        if directory:
            result = organize_files(directory)
            st.success(result)
        else:
            st.warning("Please enter a directory path!")

#-----------------------FILE ORGANIZER WEB ------------------------------
elif page == "Web File Organizer":
    st.title("File Organization Automation 📂")
    st.write("Upload a ZIP file, and we'll organize its contents into categories!")

    # File categories and extensions
    FILE_TYPES = {
        "Documents": [".pdf", ".docx", ".txt", ".xlsx", ".pptx"],
        "Images": [".jpg", ".jpeg", ".png", ".gif"],
        "Videos": [".mp4", ".mov", ".avi"],
        "Music": [".mp3", ".wav", ".flac"],
        "Archives": [".zip", ".rar", ".7z"],
        "Programs": [".exe", ".msi", ".sh"]
    }

    # Upload ZIP file
    uploaded_file = st.file_uploader("Upload a ZIP file", type=["zip"])

    if uploaded_file:
        # Path to save uploaded ZIP file
        zip_path = "temp.zip"
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract the ZIP file manually
        extract_path = "extracted_files"
        if os.path.exists(extract_path):
            for root, dirs, files in os.walk(extract_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        st.success("Files extracted successfully!")

        # Create the 'result' directory and categories manually
        result_dir = "result"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # Create subfolders for each category inside 'result'
        category_folders = {category: os.path.join(result_dir, category) for category in FILE_TYPES.keys()}
        for folder in category_folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Create "Others" folder for unrecognized files
        others_folder = os.path.join(result_dir, "Others")
        if not os.path.exists(others_folder):
            os.makedirs(others_folder)

        # Function to recursively organize files from subdirectories
        def organize_files(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    moved = False
                    for category, extensions in FILE_TYPES.items():
                        if any(file.lower().endswith(ext) for ext in extensions):
                            category_folder = category_folders[category]
                            dest_path = os.path.join(category_folder, file)
                            
                            # If file exists at destination, delete it
                            if os.path.exists(dest_path):
                                os.remove(dest_path)
                            
                            # Move the file to the appropriate folder
                            shutil.move(file_path, dest_path)
                            moved = True
                            break

                    # If no category matches, move to "Others"
                    if not moved:
                        others_dest_path = os.path.join(others_folder, file)
                        
                        # If file exists at "Others" folder, delete it
                        if os.path.exists(others_dest_path):
                            os.remove(others_dest_path)
                        
                        shutil.move(file_path, others_dest_path)

        # Organize the files when the button is clicked
        if st.button("Organize Files"):
            organize_files(extract_path)
            st.success("Files organized successfully!")

            # Now zip the 'result' folder manually
            zip_output = "result.zip"
            with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(result_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, result_dir))

            # Allow user to download the result zip file
            with open(zip_output, "rb") as f:
                st.download_button("Download Organized Files", f, file_name="result.zip")

elif page== "Database Project":
    
    # Create a SQLite database connection
    DATABASE_URL = "sqlite:///employee_management.db"
    engine = create_engine(DATABASE_URL)

    # Define the Base class and Employee model
    Base = declarative_base()

    class Employee(Base):
        __tablename__ = 'employees'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String, nullable=False)
        age = Column(Integer, nullable=False)
        department = Column(String, nullable=False)
        salary = Column(Float, nullable=False)

    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Ensure the table exists
    Base.metadata.create_all(engine)  # Create the table if it doesn't exist

    # Streamlit Interface
    st.title("Employee Management System")

    # Add a new employee
    def add_employee():
        st.subheader("Add New Employee")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=18)
        department = st.text_input("Department")
        salary = st.number_input("Salary", min_value=1000)
        
        if st.button("Add Employee"):
            if name and department:
                new_employee = Employee(name=name, age=age, department=department, salary=salary)
                session.add(new_employee)
                session.commit()
                st.success(f"Employee {name} added successfully!")
            else:
                st.warning("Please fill in all fields!")

    # View all employees
    def view_employees():
        st.subheader("View All Employees")
        employees = session.query(Employee).all()
        
        if employees:
            for employee in employees:
                st.write(f"ID: {employee.id}, Name: {employee.name}, Age: {employee.age}, Department: {employee.department}, Salary: {employee.salary}")
        else:
            st.warning("No employees found!")

    # Update employee details
    def update_employee():
        st.subheader("Update Employee Details")
        employee_id = st.number_input("Employee ID", min_value=1)
        name = st.text_input("New Name")
        age = st.number_input("New Age", min_value=18)
        department = st.text_input("New Department")
        salary = st.number_input("New Salary", min_value=1000)
        
        if st.button("Update Employee"):
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if employee:
                if name:
                    employee.name = name
                if age:
                    employee.age = age
                if department:
                    employee.department = department
                if salary:
                    employee.salary = salary
                    
                session.commit()
                st.success(f"Employee {employee_id} updated successfully!")
            else:
                st.warning("Employee not found!")

    # Delete an employee
    def delete_employee():
        st.subheader("Delete Employee")
        employee_id = st.number_input("Employee ID", min_value=1)
        
        if st.button("Delete Employee"):
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if employee:
                session.delete(employee)
                session.commit()
                st.success(f"Employee {employee_id} deleted successfully!")
            else:
                st.warning("Employee not found!")

    # Sidebar navigation
    menu = ["Add Employee", "View Employees", "Update Employee", "Delete Employee"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Add Employee":
        add_employee()
    elif choice == "View Employees":
        view_employees()
    elif choice == "Update Employee":
        update_employee()
    elif choice == "Delete Employee":
        delete_employee()

elif page == "Blog Application":
    
    # Initialize session state for blog posts if not already initialized
    if 'posts' not in st.session_state:
        st.session_state['posts'] = []

    # Function to display all blog posts
    def display_posts():
        if st.session_state['posts']:
            for idx, post in enumerate(st.session_state['posts']):
                st.write(f"### {post['title']}")
                st.write(post['content'])
                st.write(f"*Posted on: {post['date']}*")
                st.markdown("---")
                edit_button = st.button(f"Edit Post {idx+1}", key=f"edit_{idx}")
                delete_button = st.button(f"Delete Post {idx+1}", key=f"delete_{idx}")
                
                if edit_button:
                    edit_post(idx)
                
                if delete_button:
                    delete_post(idx)
        else:
            st.write("No posts available.")

    # Function to create a new blog post
    def create_post():
        st.subheader("Create a New Post")
        title = st.text_input("Title")
        content = st.text_area("Content")
        
        if st.button("Create Post"):
            if title and content:
                new_post = {
                    "title": title,
                    "content": content,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }
                st.session_state['posts'].append(new_post)
                st.success("Post created successfully!")
            else:
                st.warning("Please fill out all fields!")

    # Function to edit a blog post
    def edit_post(index):
        post = st.session_state['posts'][index]
        st.subheader("Edit Post")
        title = st.text_input("Title", post["title"])
        content = st.text_area("Content", post["content"])

        if st.button("Save Changes"):
            if title and content:
                st.session_state['posts'][index] = {
                    "title": title,
                    "content": content,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }
                st.success("Post updated successfully!")
            else:
                st.warning("Please fill out all fields!")

    # Function to delete a blog post
    def delete_post(index):
        del st.session_state['posts'][index]
        st.success("Post deleted successfully!")

    # Streamlit Interface
    st.title("Streamlit Blog Application")

    # Sidebar for navigation
    menu = ["Home", "Create Post"]
    choice = st.sidebar.selectbox("Select an option", menu)

    # Custom CSS to style the app
    st.markdown("""
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
            }

            h1 {
                text-align: center;
                margin-top: 20px;
            }

            .stButton>button {
                background-color: #007BFF;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
            }

            .stButton>button:hover {
                background-color: #0056b3;
            }

            .stTextInput>div, .stTextArea>div {
                margin-bottom: 10px;
            }

            .stMarkdown {
                margin-top: 10px;
                font-size: 14px;
                color: #555;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display content based on selected option
    if choice == "Home":
        display_posts()
    elif choice == "Create Post":
        create_post()

elif page == "ML-House Price Prediction":
    # Load the trained model
    with open('california_house_price_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Streamlit app title
    st.title("California House Price Prediction")

    # Input fields for user to enter all features
    st.subheader("Enter House Features")

    # Input fields for each of the 8 features in the dataset
    med_inc = st.number_input("Median Income (MedInc)", min_value=0.0, value=3.5)
    house_age = st.number_input("House Age (HouseAge)", min_value=0, max_value=100, value=30)
    ave_rooms = st.number_input("Average Rooms per House (AveRooms)", min_value=1, max_value=20, value=6)
    ave_occup = st.number_input("Average Occupancy (AveOccup)", min_value=0.0, max_value=10.0, value=2.0)
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=34.0)
    long = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-118.0)
    population = st.number_input("Population", min_value=0, value=1000)
    avg_rooms = st.number_input("Avg Rooms per House", min_value=1, max_value=20, value=6)

    # Create a list of features from the inputs
    features = [med_inc, house_age, ave_rooms, ave_occup, lat, long, population, avg_rooms]

    # Make prediction button
    if st.button("Predict House Price"):
        # Predict the house price using the trained model
        prediction = model.predict([features])
        ans = prediction[0]
        if ans<0:
            ans*=-1
        st.success(f"The predicted house price is ${ans:0.2f}")


elif page=="DL-Image Classification":
    # Load the pre-trained model
    model = tf.keras.models.load_model('cnn_model.h5')

    # Class labels for the CIFAR-10 dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Streamlit UI
    st.title("Image Classification with CNN")


    st.markdown("""
    This model can only detect **airplanes**, **automobiles**, **birds**, **cats**, **deers**, **dogs**, **frogs**, **horses**, **ships**, and **trucks**. 
    Please upload images from these categories only.
    """)

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)
        
        # Preprocess the image (resize and normalize)
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        image = image.reshape(1, 32, 32, 3)

        # Make a prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Show the image and prediction result
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Predicted label: {class_names[predicted_class]}")

elif page == "API-Integration Weather Dashboard":
    # Function to get the weather data from Open-Meteo API
    def get_weather_data(city):
        # Open-Meteo API URL
        base_url = "https://api.open-meteo.com/v1/forecast?"
        
        # Mapping Indian cities to their respective latitude and longitude
        city_coordinates = {
            "new delhi": (28.6139, 77.2090),      # New Delhi
            "mumbai": (19.0760, 72.8777),         # Mumbai
            "bangalore": (12.9716, 77.5946),      # Bangalore
            "kolkata": (22.5726, 88.3639),        # Kolkata
            "chennai": (13.0827, 80.2707),        # Chennai
            "hyderabad": (17.3850, 78.4867),      # Hyderabad
            "pune": (18.5204, 73.8567),           # Pune
            "ahmedabad": (23.0225, 72.5714),      # Ahmedabad
            "chandigarh": (30.7333, 76.7794),      # Chandigarh
            "jaipur": (26.9124, 75.7873),         # Jaipur
        }
        
        # Normalize city input
        city = city.lower()
        
        # Check if the city is in our dictionary
        if city in city_coordinates:
            latitude, longitude = city_coordinates[city]
        else:
            return "City not supported."
        
        # Prepare the complete URL to fetch weather data
        complete_url = f"{base_url}latitude={latitude}&longitude={longitude}&current_weather=true"
        response = requests.get(complete_url)
        data = response.json()
        
        # Check if the data contains 'current_weather'
        if "current_weather" in data:
            return data["current_weather"]
        else:
            return None

    st.title("Weather Dashboard")
    st.write("Enter an Indian city to get its current weather information.")

    # Get city input from user
    city = st.text_input("Enter City", "New Delhi")

    # Get weather data based on the input city
    data = get_weather_data(city)

    # Display the weather data
    if isinstance(data, dict):  # Check if valid weather data is returned
        temperature = data["temperature"]
        wind_speed = data["windspeed"]
        weather_desc = data["weathercode"]
        
        st.write(f"City: {city.title()}")
        st.write(f"Temperature: {temperature}°C")
        st.write(f"Wind Speed: {wind_speed} km/h")
        st.write(f"Weather Description Code: {weather_desc}")
    else:
        st.write(data)  # If city is not supported or other error

elif page=="Interactive Visualizations (Plotly+Dash)":
    # Generate some sample financial data
    date_range = pd.date_range(start="2023-01-01", periods=100, freq="D")
    stock_prices = pd.Series([100 + i + (i**0.5)*10 for i in range(100)], index=date_range)

    # Create a DataFrame for financial data
    financial_data = pd.DataFrame({'Date': date_range, 'Stock Price': stock_prices})

    # Simulate stock data for Candlestick Chart
    candlestick_data = pd.DataFrame({
        'Date': date_range,
        'Open': stock_prices + np.random.normal(0, 1, 100),
        'High': stock_prices + np.random.normal(2, 1, 100),
        'Low': stock_prices - np.random.normal(1, 1, 100),
        'Close': stock_prices + np.random.normal(0, 0.5, 100)
    })

    # Monthly returns for Bar Chart
    financial_data['Monthly Returns'] = stock_prices.pct_change().fillna(0)
    monthly_returns = financial_data['Monthly Returns'].resample('M').sum()

    # Simulated sector distribution for Pie Chart
    sectors = ['Tech', 'Healthcare', 'Finance', 'Consumer Goods', 'Energy']
    sector_distribution = [30, 25, 20, 15, 10]

    # Streamlit sidebar
    st.sidebar.header("Financial Data Overview")
    st.sidebar.markdown("""
    This dashboard displays financial data visualizations, including a candlestick chart, monthly returns bar chart, and sector distribution pie chart.
    Some data has been simulated for demonstration purposes.
    """)
    st.sidebar.markdown("### Sample Financial Data:")
    st.sidebar.write(financial_data.head())

    # Title of the main app
    st.title("Interactive Financial Data Visualizations")

    # Candlestick Chart
    st.subheader("Candlestick Chart")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=candlestick_data['Date'],
        open=candlestick_data['Open'],
        high=candlestick_data['High'],
        low=candlestick_data['Low'],
        close=candlestick_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig_candle.update_layout(
        title='Stock Price Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark'
    )
    st.plotly_chart(fig_candle)

    # Monthly Returns Bar Chart
    st.subheader("Monthly Returns")
    fig_returns = px.bar(
        monthly_returns,
        x=monthly_returns.index,
        y=monthly_returns.values,
        labels={'x': 'Month', 'y': 'Monthly Returns'},
        title=' '
    )
    st.plotly_chart(fig_returns)

    # Sector Distribution Pie Chart
    st.subheader("Simulated Sector Distribution")
    fig_pie = px.pie(
        names=sectors,
        values=sector_distribution,
        title=" "
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie)

