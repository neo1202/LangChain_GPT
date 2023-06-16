import React, { useState } from 'react'
import axios from 'axios';
import DocUploader from './DocUploader';


function App() {

	const [userInput, setUserInput] = useState('');
	const [messages, setMessages] = useState([{HumanInput:'', AiResponse:'頂級Ai在此為您服務'}]); //存儲本次聊天室對話紀錄
	//[{'AiResponse':'haha'}]

	const handleInputChange = (e) => {
		setUserInput(e.target.value);
	};

	const handleSendMessage = async (e) => {
		e.preventDefault();
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
			alert(`Error: ${err}`)
		})
	};

	return (
	<>
		<div className="absolute right-0 top-0 -z-50 h-screen w-screen bg-[url('https://source.unsplash.com/OTy0mkqc2Yk')] bg-cover bg-fixed bg-center  bg-no-repeat">
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
									<div className="flex item-center mb-2 mr-2 max-w-[65%] rounded-l-3xl rounded-tr-xl bg-gradient-to-r from-sky-500/80 to-blue-500/70 px-3 py-2 text-white">
                                    	{message.HumanInput}
                                    </div>
									<img alt="" className="h-10 w-10 rounded-full object-cover" 
										 src={"https://source.unsplash.com/pUhxoSapPFA"} 
									/>
								</div>
							}
							<div className="ml-2 mb-6 item-center flex flex-row justify-start font-bold">
								<img alt="" className="h-10 w-10 rounded-full object-cover" 
									 src={"https://source.unsplash.com/pUhxoSapPFA}"} 
								/>
								<div className="ml-2 rounded-r-3xl rounded-tl-xl bg-gray-400 px-3 py-2 text-white">
									{message.AiResponse}
								</div>
							</div>
						</div>
					))}
					
				</div>
			</div>
			<div className="mt-6 mx-auto z-0 flex h-[4rem] w-[70%] flex-row items-center rounded-xl bg-zinc-200 px-4">
				<form
					className="mx-auto flex w-full"
					onSubmit={handleSendMessage}
				>
					<input
						className="ml-4 mr-2 w-3/4 shrink rounded border border-gray-300 p-2 focus:border-blue-500 focus:outline-none"
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
