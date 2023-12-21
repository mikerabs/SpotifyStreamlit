import streamlit as st
from merge_files import gen_csv
from merge_files import merge_csv_files

with st.form("playlist_data"):
   st.write("Upload your own Spotify Top 100 songs here!")
   st.write("Since Your Top Songs 20XX is a special playlist from Spotify, we'll need to do a couple of steps before submission.")
   st.write("Step 1: Navigate to Your Top Songs 2023, do not press copy link yet!")
   st.write("Step 2: Click on the 3 dots next to the playlist, and press/click 'Add to other playlist'")
   st.write("Step 3: Press/Click 'New Playlist', and 'Create'")
   st.write("Step 4: Now, press copy link on the duplicate playlist")


   playlist_data = st.text_input('Playlist Link Here!')
   email_data = st.text_input('Email address')
   submitted = st.form_submit_button('Submit')

   if submitted:
    #Gotta do some control here just in case people hit the submit button before putting anything in

    # duplicate_email = False
    # # Get a list of all CSV files in the folder
    # csv_files = [file for file in os.listdir("spotify_files") if file.endswith('.csv')]
    # for i in range(len(csv_files)):
    #     if duplicate_email == True:
    #        break
    #     if email_data in csv_files[i]:
    #        st.write("Please do not submit duplicate playlists from the same email")
    #        duplicate_email = True
    
    # if duplicate_email == True:
    #    st.write("Please refresh and resubmit")
    # else:
    gen_csv(playlist_data, email_data, '/spotify_files/spotify'+str(email_data)+'.csv')
    merge_csv_files('','spotify_files')
