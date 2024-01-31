import streamlit as st
import pandas as pd
import pickle
from model import apply_mapping

with open("model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open('input_columns.pkl', 'rb') as file:
    input_columns = pickle.load(file)




def predictions(online_order, book_table, location, approx_cost , category , city, stacked_cuisines, stacked_rest_type):

    df = pd.DataFrame(columns=input_columns)
    df.at[0,'online_order'] = online_order
    df.at[0,'book_table'] = book_table
    df.at[0,'location'] = location
    df.at[0,' approx_cost'] =  approx_cost
    df.at[0,'category'] = category
    df.at[0,'city'] = city
    df.at[0,'stacked_cuisines'] = stacked_cuisines
    df.at[0,'stacked_rest_type'] = stacked_rest_type
    result = model.predict(df)
    return result




def main():
    st.title("Zomato App")

    

    location_options = ['Banashankari', 'Basavanagudi', 'Jayanagar', 'Kumaraswamy Layout',
        'Rajarajeshwari Nagar', 'Mysore Road', 'Uttarahalli','South Bangalore', 'Vijay Nagar', 'Bannerghatta Road', 'JP Nagar',
        'BTM', 'Wilson Garden', 'Koramangala 5th Block', 'Shanti Nagar','Richmond Road', 'City Market', 'Bellandur', 'Sarjapur Road',
        'Marathahalli', 'HSR', 'Old Airport Road', 'Indiranagar','Koramangala 1st Block', 'East Bangalore', 'MG Road',
        'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor',
        'Residency Road', 'Shivajinagar', 'Infantry Road','St. Marks Road', 'Cunningham Road', 'Race Course Road', 'Domlur',
        'Koramangala 8th Block', 'Frazer Town', 'Ejipura', 'Vasanth Nagar','Jeevan Bhima Nagar', 'Old Madras Road', 'Commercial Street',
        'Koramangala 6th Block', 'Majestic', 'Langford Town','Koramangala 7th Block', 'Brookefield', 'Whitefield',
        'ITPL Main Road, Whitefield', 'Varthur Main Road, Whitefield','Koramangala 2nd Block', 'Koramangala 3rd Block',
        'Koramangala 4th Block', 'Koramangala', 'Bommanahalli','Hosur Road', 'Seshadripuram', 'Electronic City', 'Banaswadi',
        'North Bangalore', 'RT Nagar', 'Kammanahalli', 'Hennur','HBR Layout', 'Kalyan Nagar', 'Thippasandra','CV Raman Nagar','Kaggadasapura',
        'Kanakapura Road', 'Nagawara','Rammurthy Nagar','Sankey Road','Central Bangalore', 'Malleshwaram','Sadashiv Nagar', 'Basaveshwara Nagar',
        'Rajajinagar','New BEL Road', 'West Bangalore', 'Yeshwantpur', 'Sanjay Nagar','Sahakara Nagar', 'Jalahalli', 'Yelahanka', 'Magadi Road','KR Puram']
    location = st.selectbox("Select a location:", location_options)



    category_option = ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out','Drinks & nightlife', 'Pubs and bars']
    category = st.selectbox ("select a category",category_option)



    city_option= ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
        'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
        'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
        'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
        'Koramangala 4th Blocity_optionck', 'Koramangala 5th Block',
        'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
        'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
        'Old Airport Road', 'Rajajinagar', 'Residency Road',
        'Sarjapur Road', 'Whitefield']
    city = st.selectbox("Select a city:", city_option)


    cuisines_option = ['North Indian', 'Mughlai', 'Chinese', 'Thai', 'Cafe', 'Mexican',
        'Italian', 'South Indian', 'Rajasthani', 'Pizza', 'Continental',
        'Momos', 'Beverages', 'Fast Food', 'American', 'French',
        'European', 'Burger', 'Biryani', 'Street Food', 'Rolls',
        'Ice Cream', 'Desserts', 'Andhra', 'Healthy Food', 'Salad',
        'Asian', 'Korean', 'Indonesian', 'Japanese', 'Goan', 'Seafood',
        'Kebab', 'Steak', 'Sandwich', 'Bakery', 'Vietnamese', 'Juices',
        'Arabian', 'BBQ', 'Mangalorean', 'Tea', 'Afghani', 'Finger Food',
        'Tibetan', 'Mithai', 'Middle Eastern', 'Mediterranean', 'Bengali',
        'Charcoal Chicken', 'Kerala', 'Oriya', 'Bihari', 'Roast Chicken',
        'Bohri', 'African', 'Lebanese', 'Hyderabadi', 'Belgian',
        'South American', 'Maharashtrian', 'Konkan', 'Chettinad', 'Wraps',
        'Turkish', 'Coffee', 'Afghan', 'Modern Indian', 'Iranian',
        'Lucknowi', 'Gujarati', 'Tex-Mex', 'Spanish', 'Malaysian',
        'Burmese', 'Sushi', 'Portuguese', 'Parsi', 'Nepalese', 'Greek',
        'North Eastern', 'Bar Food', 'Singaporean', 'Awadhi', 'Naga',
        'Cantonese', 'Bubble Tea', 'Kashmiri', 'Assamese', 'Sri Lankan',
        'Grill', 'British', 'Pan Asian', 'German', 'Russian', 'Jewish',
        'Vegan', 'Sindhi']



    stacked_cuisines = st.selectbox("Select a cuisines:", cuisines_option)




    restaurant_option = ['Casual Dining', 'Cafe', 'Quick Bites', 'Delivery',
        'Dessert Parlor', 'Pub', 'Beverage Shop', 'Bar', 'Takeaway',
        'Food Truck', 'Bakery', 'Sweet Shop', 'Microbrewery', 'Lounge',
        'Food Court', 'Kiosk', 'Mess', 'Club', 'Fine Dining',
        'Irani Cafee', 'Dhaba']
    stacked_rest_type = st.selectbox("Select a restaurant:", restaurant_option)


    approx_cost = st.number_input("approx cost for 2 people:", step=1)

    
    book_table = st.selectbox("Availability for booking table", ["Yes","No"])

    
    online_order = st.selectbox("Availability for online order:",  ["Yes", "No"])



    if st.button("Predict"):
        result = predictions(online_order, book_table, location, approx_cost, category, city, stacked_cuisines, stacked_rest_type)
        result_label = "Success" if result == 1 else "Failed"
        st.write(f"The prediction result is: {result_label}")




main()