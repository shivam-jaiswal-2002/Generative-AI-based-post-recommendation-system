from ai.post_recommender import PostRecommender
import streamlit as st
import json


def init():
    session_state_vars = [
        "users_tags",
        "topic_tags",
    ]
    
    for vars in session_state_vars:
        if vars not in st.session_state:
            st.session_state[vars] = None

def main():
    post_recommender = PostRecommender(tags_txt_path="./ai/tags.txt")
    
    st.title("Post Recommender")
    st.subheader("This is a post recommender system based on generative AI")
    
    # Example of how to use it in your Streamlit app
    if st.button("Get Tags from File"):
        # Load the dataset from the JSON file
        json_file_path = "./dataset/train.json"  # Update with your actual file path
        with open(json_file_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        
        # create an instance of the PostRecommender class
        with st.spinner("Loading AI model"):

            # get the user tags from the dataset
            users_tags_from_file = post_recommender.get_user_tags_from_file(dataset)
            #save users tags to session state
            st.session_state.users_tags = users_tags_from_file
            # display the user tags
    st.subheader("Users Tags from File")
    if st.session_state.users_tags:
        st.write(st.session_state.users_tags)
    #take tag input from user
    topic = st.text_input("Enter the topic")
    if st.button("Get Interested Users"):
        # create an instance of the PostRecommender class
        with st.spinner("Loading AI model"):
            wiki_text = post_recommender.summarize_doc(topic)
            tags_from_link = post_recommender.get_topic_tags(wiki_text)
            st.write(tags_from_link)
            # find and display matching users
            matching_users = post_recommender.find_matching_users(tags_from_link, st.session_state.users_tags)
            st.subheader("Matching Users")
            if matching_users:
                st.write(matching_users)
            else:
                st.write("No matching users found.")
    
    
    
   

if __name__ == "__main__":
    init()
    main()
