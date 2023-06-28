import React, { useState } from 'react';
import axios from 'axios';
import DocUploader from './DocUploader';
import human_pic from './assets/human.jpg';
import ai_pic from './assets/ai.jpg';
import bg_pic from './assets/bg.jpg';


function App() {

	const [userInput, setUserInput] = useState('');
	const [messages, setMessages] = useState([{HumanInput:'', AiResponse:'頂級Ai在此為您服務'}]); //存儲本次聊天室對話紀錄
	//[{'AiResponse':'haha'}]
	const [loading, setLoading] = useState(false);

	const handleInputChange = (e) => {
		setUserInput(e.target.value);
	};

	const handleSendMessage = async (e) => {
		e.preventDefault();
		setLoading(true);
		const text = userInput
		setUserInput('')
		axios.post('http://127.0.0.1:5000/get_answer', {
			inputText : text
		})
		.then(function (response){
			console.log(response.data.response)
			//console.log('status:', response.status)
			var status = response.status;
			const resp = status === 200 ? response.data.response : '暫時無法解析您的問題';
			//console.log('flaskRes:', resp)
			const newMessage = {
				HumanInput: userInput,
				AiResponse: resp,
			};
			//console.log('this newMessage:', newMessage);
			setMessages((prevMessages) => [...prevMessages, newMessage]);
		})
		.catch(function (err){
			alert(`Error: ${err}`);
		})
		.finally(() => {
			setLoading(false)
		});
	};

	return (
	<>
		<div className="absolute right-0 top-0 -z-50 h-screen w-screen bg-[url('./assets/bg.jpg')] bg-cover bg-fixed bg-center  bg-no-repeat">
		</div>
		<DocUploader/>
		<section className="flex flex-col items-center justify-center h-screen">
			<div className="w-4/5 h-3/4 bg-gray-200 rounded-lg container relative min-w-[450px] overflow-y-auto border-2 pb-4">
				<div className="h-[60px] bg-gray-400 mx-auto flex sticky top-0 w-full mb-8">
					<div className="text-3xl mx-auto text-center font-semibold place-self-center">
						這是跟GPT的聊天室</div>
				</div>
				<div>
					{messages.map((message, index) => (
						<div key={index}>
							{message.HumanInput && 
								<div className="mr-2 mb-2 item-center flex flex-row justify-end font-bold">
									<div style={{ whiteSpace: 'pre-wrap' }} className="flex item-center mb-2 mr-2 max-w-[65%] rounded-l-3xl rounded-tr-xl bg-gradient-to-r from-sky-500/80 to-blue-500/70 px-3 py-2 text-white">
                                    	{message.HumanInput}
                                    </div>
									<img alt="" className="h-10 w-10 rounded-full object-cover" 
										 src={human_pic} 
									/>
								</div>
							}
							<div className="ml-2 mb-6 item-center flex flex-row justify-start font-bold">
								<img alt="" className="h-10 w-10 rounded-full object-cover" 
									 src={ai_pic} 
								/>
								<div style={{ whiteSpace: 'pre-wrap' }} className="ml-2 rounded-r-3xl rounded-tl-xl bg-gray-400 px-3 py-2 text-white">
									{message.AiResponse}
								</div>
							</div>
						</div>
					))}
					{loading && (<div className="flex items-center justify-center">
						<div className="mx-auto" role="status">
							<svg aria-hidden="true" className="w-8 h-8 mr-2 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
								<path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
								<path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
							</svg>
							<span className="sr-only">Loading...</span>
						</div>
					</div> )}

				</div>
			</div>
			<div className="mt-6 mx-auto z-0 flex h-[4rem] w-[70%] flex-row items-center rounded-xl bg-zinc-200 px-4">
				<form
					className="px-6 mx-auto flex w-full"
					onSubmit={handleSendMessage}
				>
					<input
						className="ml-4 lg:ml-8 mr-2 w-3/4 shrink rounded border border-gray-300 p-2 focus:border-blue-500 focus:outline-none"
						id="content"
						placeholder="Message"
						type="text"
						value={userInput}
						onChange={handleInputChange}
					/>
					<div className="ml-4 flex flex-none items-center">
						<button
							className="flex shrink-0 items-center justify-center rounded-xl bg-indigo-500 px-4 py-1 text-white hover:bg-indigo-600"
							type="submit"
						>
							<span>Send</span>
							<span className="ml-2">
								<svg
									className="-mt-px h-4 w-4 rotate-45"
									fill="none"
									stroke="currentColor"
									viewBox="0 0 24 24"
									xmlns="http://www.w3.org/2000/svg"
								>
									<path
										d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
										strokeLinecap="round"
										strokeLinejoin="round"
										strokeWidth="2"
									/>
								</svg>
							</span>
						</button>
					</div>
				</form>
			</div>
		</section>
	</>
  	)
}

export default App
