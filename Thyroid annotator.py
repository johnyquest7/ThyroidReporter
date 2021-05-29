### Hacked together by Johnson Thomas 

### Annotation UI created in Streamlit
### Can be used for binary or multilabel annotation 

### Importing libraries
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
import streamlit as st
from PIL import Image
import os
import pandas as pd
import re

### Creating a 3 column layout in streamlit

col1, col2, col3= st.beta_columns([3, 1,1])


### Folder where the image files are kept. This path is in windows format.
### If you are runnin it in Linux, chnage the path appropriately. 
#source_dir = r'C:\Users\JOHNY\CV_recepies\cv\images' 
source_dir = None

csv_to_log = r'C:\Users\JOHNY\CV_recepies\annot1.csv'

proj_file = r'C:\Users\JOHNY\CV_recepies\St_annot.csv'


### Function to create a python list cotaning paths to image files in a specific folder
### This function is decorated with @st.cache to avoid rerunning 

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
@st.cache(allow_output_mutation=True)
def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return sorted(file_list)
	

### Creating the side bar


add_proj_text = st.sidebar.write('Start new project')
add_textbox = st.sidebar.text_input('Project name')
add_foldbox = st.sidebar.text_input('Folder name' )
add_newproj_btn = st.sidebar.button('Create new project')
st.sidebar.write(' ')

add_proj_load = st.sidebar.write('Load project')
#proj_list =new_installation(proj_file)
add_csvbox = st.sidebar.selectbox('Pick your project',"exp1")
add_loadproj_btn = st.sidebar.button('Load project')


### store file names to a list and find the number of files in the list
#file_to_anot = get_file_list(source_dir)

#file_to_anot = get_file_list(source_dir)
#max_ind= len(file_to_anot) -1


### Creating a list to store the annotations 
### @st.cache(allow_output_mutation=True) - is used to preserve the current state and to allow modification of the list
@st.cache(allow_output_mutation=True)
def init_anot(file_to_anot):
	anot = [None]*(len(file_to_anot))
	comp_list = [None]*(len(file_to_anot))
	echo_list =  [None]*(len(file_to_anot))
	shape_list =[None]*(len(file_to_anot))
	marg_list = [None]*(len(file_to_anot))
	foci_list = [None]*(len(file_to_anot))
	return anot,comp_list,echo_list,shape_list,marg_list,foci_list

### Creating a list to store just the file names	
@st.cache(allow_output_mutation=True)
def init_base_f(file_to_anot):
	base_file = [None]*(len(file_to_anot))
	return base_file
	
#anotf,comp_list,echo_list,shape_list,marg_list,foci_list = init_anot(file_to_anot)
#base_f = init_base_f(file_to_anot)

### Given an index this function converts path in the index to windows readable path 
### then load the imaeg and returns the loaded image
def get_image(ind_no,file_to_anot):
	file_name = file_to_anot[ind_no]
	im_file =re.sub("\\\\","\\\\\\\\", file_name)
	loaded_image = Image.open(im_file)
	return loaded_image

### Get just the image file name from the complete path string
def extract_basename(path):
  basename = re.search(r'[^\\/]+(?=[\\/]?$)', path)
  if basename:
    return basename.group(0)
	
def get_index(dta_ar, out_string):
	for i in range(len(dta_ar)):
		if dta_ar[i] == out_string:
			in_dex = i

	return in_dex
	

	
def main():
	
	state = _get_state()
	
	def set_index_in(in_num):
	
		
		state.comp_list[in_num] = get_index(comp_options, composition)
		state.echo_list[in_num] = get_index(echo_options, echo)
		state.shape_list[in_num]= get_index(shape_options, shape)
		state.marg_list[in_num] = get_index(margin_options, margin)
		state.foci_list[in_num]= get_index(foci_options, echogenic_foci)
	
	def update_choices(ind_num):
		''' This function collects the values of lables/tags for the next or previous image,
		then displays it in the user interface.
		This function is called each time Next or Previous button is pressed.
		'''
		if state.comp_list[ind_num] != None:
			state.comp = state.comp_list[ind_num]
		else:
			state.comp = 0
			
		if state.echo_list[ind_num] != None:
			state.echo = state.echo_list[ind_num]
		else: 
			state.echo = 0
			
		if state.shape_list[ind_num] !=None:
			state.shape =  state.shape_list[ind_num]
		else: 
			state.shape = 0
			
		if state.marg_list[ind_num] != None:
			state.margin = state.marg_list[ind_num]
		else: 
			state.margin = 0
			
		if state.foci_list[ind_num] != None:
			state.foci = state.foci_list[ind_num]
		else:
			state.foci = 0
		#print("This is from update", state.comp, state.echo, state.shape, state.margin, state.foci)
		
	# Initializing a state variable input
	if state.input == None:
		state.input = 0
		state.last_anot =0
		state.comp = 0
		state.echo = 0
		state.shape = 0
		state.margin = 0
		state.foci = 0
		
		
	# Creating the UI
	comp_options = ['cystic','spongiform', 'mixed cystic','solid']
	echo_options = ['anechoic','hyperechoic','isoechoic','hypoechoic','very hypoechoic']
	shape_options =['wider than tall','taller than wide']
	margin_options = ['smooth','ill defined','lobulated', 'irregular', 'ete']
	foci_options = ['none','comet tail artifacts','macrocalcifications','peripheral calcifications','punctate echogenic foci']

	with col2:
		prev_button = st.button('Previous')
		if state.active_project == True:
			composition = st.radio('Composition',comp_options, state.comp)
			echo = st.radio('Echogenicity',echo_options, state.echo)
			shape = st.radio('Shape',shape_options, state.shape)
			state.started = True 
		
	with col3:
		next_button = st.button('Next')
		if state.active_project == True:
			margin = st.radio('Margin',margin_options, state.margin)
			echogenic_foci = st.radio('Echogenic Foci', foci_options, state.foci)
			
					
	with col1:
		#if state.input ==0:
		if next_button and state.active_project == True:	
			if state.input == state.max_ind:
				e =RuntimeError('Reached end of images in the folder')
				st.exception(e)
			else:
				set_index_in(state.input)
				#update_choices(state.input,comp_list)
				state.input = state.input + 1
				update_choices(state.input)
				if state.input > state.last_anot:
					state.last_anot = state.input
		if prev_button and state.active_project == True:
			if state.input == 0:
				e =RuntimeError('Reached the first image in the folder')
				st.exception(e)
			else:
				set_index_in(state.input)
				#update_choices(state.input,state.comp_list)
				state.input = state.input -1
				update_choices(state.input)
				
		if add_newproj_btn and add_foldbox != "":
			state.file_to_anot = get_file_list(add_foldbox)
			state.max_ind= len(state.file_to_anot) -1
			### initializing variables
			state.active_project = True 
			state.input = 0
			state.last_anot =0
			state.comp = 0
			state.echo = 0
			state.shape = 0
			state.margin = 0
			state.foci = 0
			state.started = False
			state.anot_list,state.comp_list,state.echo_list,state.shape_list,state.marg_list,state.foci_list = init_anot(state.file_to_anot)
			state.base_f = init_base_f(state.file_to_anot)
			
		if add_foldbox != ""  and state.started == True:
			st.image(get_image(state.input,state.file_to_anot),use_column_width=True)	
			desc_nod, lbl, fln= gen_desc_save(composition, echo, shape, margin, echogenic_foci,state.input,state.file_to_anot )
			#print("anot list",state.anot_list)
			state.anot_list[state.input] = lbl
			state.base_f[state.input]	= fln	
			col1.write( desc_nod)
	

	### Save button  ########################################################
	
	save_button = st.button('Save')
	
	if save_button:
		set_index_in(state.input)
		df = pd.DataFrame(list(zip(state.base_f, state.anot_list)), columns =["IM_FILENAME", "LABELS"])
		cwd = os.getcwd()
		csv_to_log =  r'C:\Users\JOHNY\CV_recepies\annot1.csv'
		#print("printing curr file name")
		#print(csv_to_log)
		df.to_csv(csv_to_log)
		#proj = pd.read_csv(proj_file)
		#ind_pr= proj.index[proj['Project_name'] == curr_proj_name].tolist()
		print(ind_pr)

	state.sync()
	
def gen_desc_save(composition, echo, shape, margin, echogenic_foci, ind_no,file_to_anot):
	comp = composition.capitalize()
	if echogenic_foci =="none":
		echo_foc = "no calcification or comet tail artiacts"
	else:
		echo_foc = echogenic_foci
	desc = comp + " " + echo + " "  + shape + " thyroid nodule with " + margin + " margin"  + " and " + echo_foc + "."
	file_name2 = file_to_anot[ind_no]
	file_only = extract_basename(file_name2)
	label_to_log = composition + "," + echo + "," + shape + "," + margin + "," + echogenic_foci
	#anotf[ind_no] = label_to_log

	return desc,label_to_log, file_only


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
	main()
    #main(file_to_anot, anotf, base_f,comp_list,echo_list,shape_list,marg_list,foci_list)
