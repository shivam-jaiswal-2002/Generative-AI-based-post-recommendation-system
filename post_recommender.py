from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
import json
from dotenv import load_dotenv

load_dotenv()


class PostRecommender:
    def __init__(self, model_name="gpt-3.5-turbo", tags_txt_path="tags.txt"):
        self.model_name = model_name
        self.tags_txt_path = tags_txt_path
        self.all_tags = self.load_tags()

    def load_tags(self):
        tags = []
        with open(self.tags_txt_path, "r") as f:
            for line in f.readlines():
                tags.append(line.strip())
        return tags 

    class Tags(BaseModel):
        username: str = Field(description="username of the user. key should be [username]")
        tags: list = Field(
            description="python list of tags that the user might be intrested in key should be [tags]"
        )

    def get_user_tags(self, username, posts):
        parser = JsonOutputParser(pydantic_object=self.Tags)

        tags_prompt = """you are a social media expert , you study social media data
        i have a user and the post he likes , i want to classify the users intrest based on the post he likes into tags
        give atleast 10 tags that the user might be intrested in
        tags should be strictly from this list dont use any other tags
        output should have atleast 20 tags
        tag list : {tags}
        user name : {user_name}
        tags : {posts}
        output format should be in json format without any code blocks
        output format should be strictly followed

        """

        prompt = PromptTemplate(
            template=tags_prompt,
            input_variables=["user_name", "posts", "tags"],
            output_parser=parser,
            partial_variables={
                "format_instruction": parser.get_format_instructions(),
            },
        )

        model = ChatOpenAI(model=self.model_name)
        chain = prompt | model | parser
        output = chain.invoke(
            {"user_name": username, "posts": posts, "tags": self.all_tags}
        )
        return output

    def get_topic_tags(self, topic):
        parser = JsonOutputParser(pydantic_object=self.Tags)

        tags_prompt = """i have post you need to take out keywords or tags from the sentence and match that if any similar tags are present in the list {tags} then display those tags . dont use any other tags only display the tags that have similar meaning or somehow related to the tags obtained from user topic given
        Display atleast 5 tags .
        Remeber that tags you are displaying should be related to the tags obtained from topic : {topic} . The tags should be displayed only if they are related to the tags that given by user in topic .
        output format should be in json format without any code blocks
        and remeber to put all the tags direclty in the list like suppose if tags obtained are "finance", "cricket","singing"
        then it should be saved as a list like:
        Remeber dont make a 2d list , it should be one dimension list containing all the tags like given below . follow the exact format
        [
            "finance",
            "cricket",
            "singing"
        ]
        please follow the format above
    
        """

        prompt = PromptTemplate(
            template=tags_prompt,
            input_variables=["topic","tags"],
            output_parser=parser,
            partial_variables={
                "format_instruction": parser.get_format_instructions(),
            },
        )

        model = ChatOpenAI(model=self.model_name)
        chain = prompt | model | parser
        output = chain.invoke(
            {"topic" : topic , "tags": self.all_tags}
        )
        return output

    def recommend_users(self, input_tags, users_tags):
        matching_users = []

        # Iterate through each user and their tags in the dataset
        for username, user_tags in users_tags.items():
            # Check if any tag from the input is present in the user's tags
            if any(tag in user_tags for tag in input_tags):
                matching_users.append(username)

        return matching_users

    
    
    def get_user_tags_from_file(self, dataset):
        users_tags = {}

        for username, user_posts in dataset.items():
            user_tags_output = self.get_user_tags(username, user_posts)
            print(user_tags_output)
            print(user_tags_output['tags'])
            users_tags[username] = user_tags_output['tags']
            
        return users_tags
    
    def wiki_loader(self,topic,length=1000):
        from langchain_community.document_loaders import WikipediaLoader

        docs = WikipediaLoader(query=topic, load_max_docs=1).load()
        print(len(docs))
        print(docs[0])
        print(type(docs[0]))
        # text = docs[0].page_content[:length] # a content of the Document
        return docs
        
    def summarize_doc(self,topic):
        from langchain.chains.summarize import load_summarize_chain
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_openai import ChatOpenAI

        docs = self.wiki_loader(topic)
        print("str",docs)
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
        chain = load_summarize_chain(llm, chain_type="stuff")

        ans = chain.run(docs)
        return ans
    
    def find_matching_users(self,tags_from_link, users_tags):
        matching_users = []

        # Iterate through each user and their tags in the dataset
        for username, user_tags in users_tags.items():
            # Check if any tag from the input is present in the user's tags
            if any(tag in user_tags for tag in tags_from_link):
                matching_users.append(username)

        return matching_users
