"use strict";(self.__LOADABLE_LOADED_CHUNKS__=self.__LOADABLE_LOADED_CHUNKS__||[]).push([[34973],{603239:e=>{var t={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"CarouselEllipsis_pin",selections:[{alias:null,args:null,kind:"ScalarField",name:"entityId",storageKey:null},{args:null,kind:"FragmentSpread",name:"CarouselEllipsis_pin2"}],type:"Pin",abstractKey:null};t.hash="a4b0b28d3f9f52a7e3d5874c94bfb63d",e.exports=t},822423:e=>{var t={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"CarouselEllipsis_pin2",selections:[{args:null,kind:"FragmentSpread",name:"useLogSwipe_pin"},{args:null,kind:"FragmentSpread",name:"usePinCarouselData_pin"}],type:"Pin",abstractKey:null};t.hash="3286ed8ff7f456e30ce44b879fb3e273",e.exports=t},348853:e=>{var t,n={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"ContextMenuClickthroughLogging_pin",selections:[{alias:null,args:null,concreteType:"Board",kind:"LinkedField",name:"board",plural:!1,selections:[{alias:null,args:null,kind:"ScalarField",name:"url",storageKey:null}],storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"entityId",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"link",storageKey:null},{alias:null,args:null,concreteType:"User",kind:"LinkedField",name:"pinner",plural:!1,selections:[{alias:null,args:null,kind:"ScalarField",name:"username",storageKey:null}],storageKey:null},{alias:null,args:null,concreteType:"RichPinDataView",kind:"LinkedField",name:"richMetadata",plural:!1,selections:[{alias:null,args:null,concreteType:"ArticleMetadata",kind:"LinkedField",name:"article",plural:!1,selections:t=[{alias:null,args:null,kind:"ScalarField",name:"__typename",storageKey:null}],storageKey:null},{alias:null,args:null,concreteType:"RichPinProductMetadata",kind:"LinkedField",name:"products",plural:!0,selections:t,storageKey:null},{alias:null,args:null,concreteType:"RecipeMetadata",kind:"LinkedField",name:"recipe",plural:!1,selections:t,storageKey:null}],storageKey:null},{alias:null,args:null,concreteType:"StoryPinData",kind:"LinkedField",name:"storyPinData",plural:!1,selections:t,storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"storyPinDataId",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"trackedLink",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"trackingParams",storageKey:null},{args:null,kind:"FragmentSpread",name:"useGetStringifiedCommerceAuxData_pin"}],type:"Pin",abstractKey:null};n.hash="91958ad6d73597b90dd099ea462eb40e",e.exports=n},639920:e=>{var t,n={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"DebugSignalsFeedback_pin",selections:[{alias:null,args:null,concreteType:"SignalDecisionDict",kind:"LinkedField",name:"debAds",plural:!0,selections:t=[{alias:null,args:null,kind:"ScalarField",name:"backgroundColor",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"iconUrl",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"signalId",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"signalMessage",storageKey:null}],storageKey:null},{alias:null,args:null,concreteType:"SignalDecisionDict",kind:"LinkedField",name:"debContentQuality",plural:!0,selections:t,storageKey:null},{alias:null,args:null,concreteType:"SignalDecisionDict",kind:"LinkedField",name:"debInclusiveProduct",plural:!0,selections:t,storageKey:null},{alias:null,args:null,concreteType:"SignalDecisionDict",kind:"LinkedField",name:"debShopping",plural:!0,selections:t,storageKey:null},{alias:null,args:null,concreteType:"SignalDecisionDict",kind:"LinkedField",name:"debTrustAndSafety",plural:!0,selections:t,storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"entityId",storageKey:null}],type:"Pin",abstractKey:null};n.hash="06530ceab8fc1bb7b5d868c8ffeb2afd",e.exports=n},270643:e=>{var t={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"useLogSwipe_pin",selections:[{args:null,kind:"FragmentSpread",name:"useGetStringifiedCommerceAuxData_pin"}],type:"Pin",abstractKey:null};t.hash="dbfca9820e0aa1302554a0137a270b16",e.exports=t},413614:(e,t,n)=>{n.d(t,{Z:()=>f});var l,i=n(667294);n(167912);var a=n(616550),r=n(768559),s=n(558068),o=n(407043),d=n(186656),c=n(916117),u=n(623568),h=n(692627),p=n(999018),g=n(785893);let _=void 0!==l?l:l=n(348853);function f({children:e,hovered:t,pinKey:n,slotIndex:l,trafficSource:f,trackingParamsMap:m,viewType:x}){let{logContextEvent:y}=(0,o.v)(),[b,v]=(0,i.useState)(),[P,j]=(0,i.useState)(),k=(0,c.Z)(_,n),{entityId:w,trackedLink:S,link:D}=k,C=S||D||"",z=()=>{v(!0)},A=function(e){let t=(0,a.TH)(),{previous:n}=(0,s.Hv)();return(0,r.Z)({...e,location:t,previousHistory:n})}({boardUrl:k.board?.url,pinId:w,pinnerUserName:k.pinner?.username,storyPinDataId:k.storyPinDataId,trackingParams:k.trackingParams,trackingParamsMap:m}),I=(0,h.Z)({hasPin:!!k,hasPinRichMetadata:!!k.richMetadata,hasPinRichMetadataProducts:!!k.richMetadata?.products,hasPinRichMetadataArticle:!!k.richMetadata?.article,hasPinRichMetadataRecipe:!!k.richMetadata?.recipe,hasPinStoryPinData:!!k.storyPinData}),Z=(0,p.Z)(k),E=()=>{let e=Z();(0,d.Z)({url:"/v3/offsite/",data:{check_only:!1,pin_id:w,url:C,client_tracking_params:A,aux_data:JSON.stringify({clickthrough_type:"rightClick",objectId:w,...l||{},...e})}}).then(t=>{t&&(y({event_type:12,object_id_str:w,view_type:x,view_parameter:I,aux_data:{clickthrough_type:"rightClick",...l||{},...e}}),y({event_type:8948,view_type:x,object_id_str:w,view_parameter:I,aux_data:{click_type:"clickthrough",closeup_navigation_type:f&&(0,u.sV)(f)?"deeplink":"click",clickthrough_type:"rightClick",...l||{},...e}}))})},F=e=>{b&&(/^\/pin/.test(e.target.activeElement.attributes.href?.value)||(E(),v(!1)),window.removeEventListener(P,F,!1))};return(0,i.useEffect)(()=>{void 0!==window?.document?.hidden?j("visibilitychange"):void 0!==window?.document?.msHidden?j("msvisibilitychange"):void 0!==window?.document?.webkitHidden&&j("webkitvisibilitychange")},[]),(0,i.useEffect)(()=>(b&&window&&window.addEventListener(P,F,!1),()=>window.removeEventListener(P,F)),[b,P]),(0,i.useEffect)(()=>(t&&window.addEventListener("contextmenu",z),()=>{window.removeEventListener("contextmenu",z)}),[t]),(0,g.jsx)(i.Fragment,{children:e})}},987765:(e,t,n)=>{n.d(t,{t:()=>m,Z:()=>y});var l=n(883119),i=n(667294),a=n(391254),r=n(785893);let s="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",o=()=>[,,,,,].fill().map(()=>s[Math.floor(Math.random()*s.length)]).join(""),d=e=>{let t=[],n=Math.floor(3*Math.random()+1);for(let l=0;l<n-1;l+=1){let n=Math.floor(Math.random()*(e.length-1)+1);t.push(n)}t.sort((e,t)=>e-t);let l=[],i=0;for(let n=0;n<t.length;n+=1)l.push(e.slice(i,t[n])),i=t[n];return l.push(e.slice(i)),[...l]};function c({children:e}){let t=(0,i.useMemo)(()=>d(e.replace("  "," ")).map(e=>({className:o(),content:e})),[e]);return(0,r.jsxs)(i.Fragment,{children:[(0,r.jsx)(a.Z,{unsafeCSS:t.map(({className:e,content:t})=>`.${e} { display: inline; overflow-wrap: unset; } .${e}:before { content: "${t}"; font-weight: 600;  }`).join(" ")}),t.map(({className:e})=>(0,r.jsx)("div",{className:e},e))]})}var u=n(773285),h=n(780280),p=n(19121);let g=e=>"Promoted by"===e,_=()=>{let e=(0,p.Z)(),{checkExperiment:t}=(0,u.F)(),n=t("web_obfuscate_ads_with_global_css").group.split("_")[3]??"op";return`${n}${e.isAuth?e.id:"unauth"}`},f=()=>(0,r.jsx)("div",{className:_()}),m=()=>{let{checkExperiment:e}=(0,u.F)(),t=_();return e("web_obfuscate_ads_with_global_css").anyEnabled?(0,r.jsx)(a.Z,{unsafeCSS:`.${t}:before { content: "Promoted by"; font-weight: 600;  }`}):null};function x({children:e}){let{checkExperiment:t}=(0,u.F)();return t("web_obfuscate_ads_with_global_css").anyEnabled?(0,r.jsx)(f,{}):(0,r.jsx)(c,{children:e})}function y({children:e,lineClamp:t,size:n,underline:i,weight:a}){let{isAuthenticated:s}=(0,h.B)(),o=s&&g(e);return(0,r.jsx)(l.xv,{lineClamp:t,size:n,title:o?void 0:e,underline:i,weight:a,children:o?(0,r.jsx)(x,{children:e}):e})}},221263:(e,t,n)=>{n.d(t,{Z:()=>l});function l(e,t){return e.isAuth&&t===e.id}},937330:(e,t,n)=>{n.d(t,{Z:()=>l});function l(e,t){return!!(t.isAuth&&e)}},4294:(e,t,n)=>{n.d(t,{Z:()=>a});var l=n(221263),i=n(937330);function a(e,t,n){return(0,l.Z)(n,t)||(0,i.Z)(e,n)}},444445:(e,t,n)=>{n.d(t,{$H:()=>r,$q:()=>x,D6:()=>d,KN:()=>b,Lo:()=>i,P2:()=>_,Wv:()=>y,ZZ:()=>u,g5:()=>h,jC:()=>s,lX:()=>g,nW:()=>p,oX:()=>f,qG:()=>m,tG:()=>a,yF:()=>l,yc:()=>c,yt:()=>o});let l=236,i=2*l/3,a=175,r=24,s=4,o=8,d=2,c=2,u=14,h=16,p=12,g=16,_=24,f=16,m=-1,x=(e=!1,t=!1)=>e?t?g:p:_,y=({contentVisibleItemCount:e,gap:t,width:n})=>e||n?(n-(e-1)*t)/e:l,b=(e,t,n,l,i=u)=>{let a=e+i,r=`
@media (min-width: ${t*a}px) and (max-width: ${(n+1)*a-1}px) {
  ${l}
}
`;return r}},329734:(e,t,n)=>{n.d(t,{P:()=>a,Z:()=>i});var l=n(702664);function i(e){return e&&e[1000069]&&500417===e[1000069].experience_id&&e[1000069].display_data.hide_footer}function a(){let e=(0,l.useSelector)(({experiences:e})=>e);return!!e&&i(e)}},499659:(e,t,n)=>{n.d(t,{Q6:()=>u,ZP:()=>o,qe:()=>d,yU:()=>c});var l=n(239745);let i=(e,t)=>e.length===t.length&&e.every((e,n)=>e===t[n]),a=e=>e;function r(e,t=i,n=a){return function(l){let i=[];return function(...a){let r=i.find(e=>t(e.args,n(a)));if(r)return r.result;let s=l(...a);return i.push({args:n(a),result:s}),e&&i.length>e&&i.shift(),s}}}let s=r(),o=s,d=r(1),c=r(void 0,i,e=>[JSON.stringify(e)]),u=r(0,(e,t)=>e.length===t.length&&e.every((e,n)=>(0,l.Z)(e,t[n])))},750391:(e,t,n)=>{n.d(t,{G7:()=>d,Nl:()=>u,WE:()=>o,l6:()=>c,rh:()=>s});var l=n(27255),i=n(451820),a=n(66699);let r=e=>e.checkExperiment("web_ps4p").anyEnabled;function s(e,t,n,i,a){if(!t.isAuth)return!1;let{creatorAnalytics:s}=e,o=!!e.storyPinDataId,d=!!e.videos?.videoList,c=(a||!(o||d))&&0!==Object.keys(e.aggregatedPinData?.creatorAnalytics?._30dRealtime||{}).length,u=e.board?.privacy===l.Z.BoardPrivacy.PUBLIC,h=e.board?.privacy===l.Z.BoardPrivacy.PROTECTED,p=(h||u)&&!e.isRepin&&[t.id,i].includes(e.pinner?.entityId);return c||0!==Object.keys(s?._30dRealtime||{}).length||t.isPartner&&p||r(n)&&o&&p}function o(e,t,n,l,i,a){return e.isAuth&&!!t&&(n||l)&&!!i?.isLoaded&&!a}function d(e,t){return(0,a.Z)(t.nbt(["The last day", "Last {{ count }} days"], e, "Previous {{ count }} days from current date", true),{count:e})}function c(e,t){return(0,a.Z)(t.nbt(["Last {{ count }} hour", "Last {{ count }} hours"], e, "analytics.utils", true),{count:e})}function u({i18n:e,isRealtime:t,humanizedTimeSinceLastUpdate:n,displayLifetime:l,numDays:r,selectedDateRange:s,locale:o}){let d=[];if(r){let t=l?(0,a.Z)(e.nbt(["Percentage changes are compared with {{n}} day before {{startDate}} – {{endDate}}.", "Percentage changes are compared with {{n}} days before {{startDate}} – {{endDate}}."], r, "pinstats.toplineMetrics.description", true),{n:r,startDate:(0,i.Z)(o,s.startDate,i.k.NUMERIC),endDate:(0,i.Z)(o,s.endDate,i.k.NUMERIC)}):(0,a.Z)(e.nbt(["Percentage changes are compared with {{n}} day before the selected date range.", "Percentage changes are compared with {{n}} days before the selected date range."], r, "pinstats.toplineMetrics.description", true),{n:r});d.push(t)}return t?d.push(e.bt("Metrics are updated in real time.", "Metrics are updated in real-time.", "pinstats.PinnerToplineMetrics.description", undefined, true)):n&&d.push((0,a.Z)(e.bt("Metrics updated {{ timeSince }}.", "Metrics updated {{ timeSince }}.", "analytics.header.disclaimer.metricsUpdated", undefined, true),{timeSince:n})),d.join(" ")}},692627:(e,t,n)=>{n.d(t,{Z:()=>l});function l({hasPin:e,hasPinRichMetadata:t,hasPinRichMetadataProducts:n,hasPinRichMetadataArticle:l,hasPinRichMetadataRecipe:i,hasPinStoryPinData:a}){if(e){if(t)return n?144:l?141:i?145:139;if(a)return 157}return 140}},53325:(e,t,n)=>{n.d(t,{GZ:()=>l,OE:()=>i,zX:()=>a});let l=246,i=197,a=236},46584:(e,t,n)=>{n.d(t,{Z:()=>c});var l=n(667294);let i=new Map,a=null,r=e=>{e.forEach(e=>{let t=i.get(e.target);t&&t(e)})},s=e=>{a.unobserve(e),i.delete(e)},o=(e,t="-64px 0px 0px 0px",n)=>{let l={root:"undefined"==typeof document?null:document.querySelector("#mainContainer"),rootMargin:t,threshold:[0,.5,1]};a=a||new window.IntersectionObserver(r,l),i.set(e,n),a.observe(e)},d=e=>i.has(e);function c({onVisibilityChanged:e,inAdsDesktopVideoExperiment:t,trackFullVisible:n,rootMargin:a}){let r=(0,l.useRef)(null),c=!1,u=t=>{let n=t.intersectionRatio>0||t.isIntersecting;(c=n)&&e(!0)},h=()=>{r.current instanceof HTMLElement&&d(r.current)&&c&&(e(!1),c=!1)},p=(0,l.useCallback)(e=>{r.current instanceof HTMLElement&&o(r.current,e,e=>{if(!i.has(r.current))return;let l=t?e.intersectionRatio>=.5:e.intersectionRatio>0||e.isIntersecting,a=n?e.intersectionRatio>=1:l,s=n?0===e.intersectionRatio:!a;!c&&a?u(e):c&&s&&h()})},[r.current]);return(0,l.useEffect)(()=>(p(a),()=>{r.current instanceof HTMLElement&&(h(),s(r.current))}),[p]),r}},841509:(e,t,n)=>{n.d(t,{Z:()=>a});var l=n(883119),i=n(785893);function a({children:e,additionalStyles:t={}}){return(0,i.jsx)(l.xu,{bottom:!0,dangerouslySetInlineStyle:{__style:{pointerEvents:"none",...t}},"data-test-id":"contentLayer",left:!0,position:"absolute",right:!0,top:!0,children:e})}},674462:(e,t,n)=>{n.d(t,{Z:()=>w});var l,i=n(667294);n(167912);var a=n(616550),r=n(883119),s=n(319915),o=n(916117),d=n(241244),c=n(947599),u=n(215292),h=n(19121),p=n(785893);let g=({signal:e,anchor:t,setShowFlyout:n,setStore:l})=>{let{hovered:i,onMouseEnter:a,onMouseLeave:s}=(0,u.Z)(),o=e=>{n(e=>!e),l(e)};return(0,p.jsx)(r.iP,{onMouseEnter:a,onMouseLeave:s,onTap:()=>o(e),rounding:"pill",children:(0,p.jsxs)(r.xu,{ref:t,alignItems:"center",color:i?"dark":"transparentDarkGray",display:"flex",padding:3,rounding:"pill",children:[(0,p.jsx)(r.xu,{alignItems:"center",color:"default",display:"flex",height:24,justifyContent:"center",marginEnd:1,minWidth:24,rounding:"circle",children:(0,p.jsx)(r.xu,{height:20,overflow:"hidden",rounding:"circle",width:20,children:(0,p.jsx)(r.Ee,{alt:"",color:e.backgroundColor??"",naturalHeight:1,naturalWidth:1,src:e.iconUrl??""})})}),(0,p.jsx)(r.xv,{color:"inverse",size:"200",weight:"bold",children:e.signalMessage})]})})};var _=n(186656),f=n(898781),m=n(343341),x=n(499128);function y(e){let t=(0,f.ZP)(),{onHide:n}=e,l=(0,p.jsx)(r.xv,{children:t.bt("Thank you for helping improve signal detection on Pinterest.", "Thank you for helping improve signal detection on Pinterest!", "pinRep.actionBar.signalDetector.toast", undefined, true)});return(0,p.jsx)(x.ZP,{onHide:n,text:l})}let b=function({anchor:e,onDismiss:t,pinId:n,signal:l}){let i=(0,f.ZP)(),a=(0,h.Z)(),{showOneToast:s}=(0,m.F9)();if(!a.isEmployee)return null;let{backgroundColor:o,iconUrl:d,signalId:c,signalMessage:u}=l;return(0,p.jsx)(r.mh,{children:(0,p.jsx)(r.J2,{anchor:e,idealDirection:"right",onDismiss:t,positionRelativeToAnchor:!1,size:"lg",children:(0,p.jsxs)(r.xu,{padding:4,width:"100%",children:[(0,p.jsx)(r.kC,{alignItems:"center",justifyContent:"start",children:(0,p.jsx)(r.X6,{size:"400",children:i.bt("Signal detected:", "Signal detected:", "pinRep.actionBar.signalDetector.header", undefined, true)})}),(0,p.jsx)(r.xu,{paddingY:3,children:(0,p.jsxs)(r.kC,{alignItems:"center",gap:2,justifyContent:"center",children:[(0,p.jsx)(r.xu,{height:40,overflow:"hidden",rounding:"circle",width:40,children:d?(0,p.jsx)(r.Ee,{alt:i.bt("Signal display", "Signal display", "pinRep.actionBar.signalDetector.signalAsset", undefined, true),color:"white",naturalHeight:1,naturalWidth:1,src:d}):(0,p.jsx)(r.xu,{dangerouslySetInlineStyle:{__style:{backgroundColor:o??""}},height:40,width:40})}),(0,p.jsx)(r.xv,{size:"200",children:u})]})}),(0,p.jsx)(r.xu,{marginBottom:3,children:(0,p.jsx)(r.xv,{size:"200",children:i.bt("If you think this detection is inaccurate, please send this Pin for review. It will help improve Pinterest.", "If you think this detection is inaccurate, please send this Pin for review. It will help improve Pinterest.", "pinRep.actionBar.signalDetector.instructions", undefined, true)})}),(0,p.jsxs)(r.kC,{alignItems:"center",gap:2,justifyContent:"start",children:[(0,p.jsx)(r.zx,{color:"red",fullWidth:!0,onClick:()=>{(0,_.Z)({url:`/v3/pins/${n}/signal_request_review/`,method:"POST",data:{signal_id:c,user:a.id}}),t(),s(({hideToast:e})=>(0,p.jsx)(y,{onHide:e}))},size:"md",text:i.bt("Send for review", "Send for review", "pinRep.actionBar.signalDetector.button.review", undefined, true)}),(0,p.jsx)(r.zx,{color:"gray",fullWidth:!0,onClick:t,size:"md",text:i.bt("Cancel", "Cancel", "pinRep.actionBar.signalDetector.button.cancel", undefined, true)})]})]})})})};function v({signals:e,pinId:t}){let[n,l]=(0,i.useState)(!1),a=(0,i.useRef)(null),[o,u]=(0,i.useState)(e[0]);return(0,p.jsxs)(i.Fragment,{children:[(0,p.jsx)(s.Z,{name:"SafeSuspense_DebugSignal_FeedbackButton",children:(0,p.jsx)(c.Z,{children:(0,p.jsx)(r.xu,{"data-test-id":"debug-signals-feedback-button",children:(0,p.jsx)(d.Z,{children:(0,p.jsx)(r.kC,{alignItems:"stretch",justifyContent:"start",children:e.map(e=>(0,p.jsx)(r.xu,{marginEnd:2,children:(0,p.jsx)(g,{anchor:a,setShowFlyout:l,setStore:u,signal:e})},e.signalId))})})})})}),n&&(0,p.jsx)(b,{anchor:a.current,onDismiss:()=>l(!1),pinId:t,signal:o})]})}let P=function({anchor:e,onDismiss:t,pinId:n,signals:l}){let a=(0,f.ZP)(),[s,o]=(0,i.useState)(0),d=(0,h.Z)(),{showOneToast:c}=(0,m.F9)();if(!d.isEmployee)return null;let u=l.map(({signalMessage:e})=>({href:"#",text:e??""})),{backgroundColor:g,iconUrl:x,signalId:b,signalMessage:v}=l[s],P=({activeTabIndex:e})=>{o(e)};return(0,p.jsx)(r.mh,{children:(0,p.jsx)(r.J2,{anchor:e,idealDirection:"right",onDismiss:t,positionRelativeToAnchor:!1,size:"lg",children:(0,p.jsxs)(r.xu,{padding:4,width:"100%",children:[l.length>1&&(0,p.jsx)(r.xu,{marginBottom:4,overflow:"scrollX",padding:1,children:(0,p.jsx)(r.mQ,{activeTabIndex:s,onChange:({event:e,activeTabIndex:t,dangerouslyDisableOnNavigation:n})=>{e.preventDefault(),n(),P({activeTabIndex:t})},tabs:u})}),(0,p.jsx)(r.kC,{alignItems:"center",justifyContent:"start",children:(0,p.jsx)(r.X6,{size:"400",children:a.bt("Signal detected:", "Signal detected:", "pinRep.actionBar.signalDetector.header", undefined, true)})}),(0,p.jsx)(r.xu,{paddingY:3,children:(0,p.jsxs)(r.kC,{alignItems:"center",gap:{row:2,column:0},justifyContent:"center",children:[(0,p.jsx)(r.xu,{height:40,overflow:"hidden",rounding:"circle",width:40,children:x?(0,p.jsx)(r.Ee,{alt:a.bt("Signal display", "Signal display", "pinRep.actionBar.signalDetector.signalAsset", undefined, true),color:"white",naturalHeight:1,naturalWidth:1,src:x}):(0,p.jsx)(r.xu,{dangerouslySetInlineStyle:{__style:{backgroundColor:g??""}},height:40,width:40})}),(0,p.jsx)(r.xv,{size:"200",children:v})]})}),(0,p.jsx)(r.xu,{marginBottom:3,children:(0,p.jsx)(r.xv,{size:"200",children:a.bt("If you think this detection is inaccurate, please send this Pin for review. It will help improve Pinterest.", "If you think this detection is inaccurate, please send this Pin for review. It will help improve Pinterest.", "pinRep.actionBar.signalDetector.instructions", undefined, true)})}),(0,p.jsxs)(r.kC,{alignItems:"center",gap:{row:2,column:0},justifyContent:"start",children:[(0,p.jsx)(r.zx,{color:"red",fullWidth:!0,onClick:()=>{(0,_.Z)({url:`/v3/pins/${n}/signal_request_review/`,method:"POST",data:{signal_id:b,user:d.id}}),t(),c(({hideToast:e})=>(0,p.jsx)(y,{onHide:e}))},size:"md",text:a.bt("Send for review", "Send for review", "pinRep.actionBar.signalDetector.button.review", undefined, true)}),(0,p.jsx)(r.zx,{color:"gray",fullWidth:!0,onClick:t,size:"md",text:a.bt("Cancel", "Cancel", "pinRep.actionBar.signalDetector.button.cancel", undefined, true)})]})]})})})},j=void 0!==l?l:l=n(639920);function k({signals:e,pinId:t}){let{backgroundColor:n,iconUrl:l,signalMessage:a}=e[0],[o,h]=(0,i.useState)(!1),g=(0,i.useRef)();return(0,p.jsxs)(i.Fragment,{children:[(0,p.jsx)(s.Z,{name:"SafeSuspense_DebugSignal_FeedbackButton",children:(0,p.jsx)(c.Z,{children:(0,p.jsx)(r.xu,{"data-test-id":"debug-signals-feedback-button",children:(0,p.jsx)(d.Z,{children:(0,p.jsx)(u.q,{children:({hovered:e,onMouseEnter:t,onMouseLeave:i})=>(0,p.jsx)(r.iP,{onMouseEnter:t,onMouseLeave:i,onTap:()=>h(e=>!e),rounding:"pill",children:(0,p.jsxs)(r.xu,{ref:g,alignItems:"center",color:e?"dark":"transparentDarkGray",display:"flex",padding:3,rounding:"pill",children:[(0,p.jsx)(r.xu,{alignItems:"center",color:"default",display:"flex",height:24,justifyContent:"center",marginEnd:1,rounding:"circle",width:24,children:(0,p.jsx)(r.xu,{height:20,overflow:"hidden",rounding:"circle",width:20,children:(0,p.jsx)(r.Ee,{alt:"",color:n??"",naturalHeight:1,naturalWidth:1,src:l??""})})}),(0,p.jsx)(r.xv,{color:"inverse",size:"200",weight:"bold",children:a})]})})})})})})}),o&&g&&g.current&&(0,p.jsx)(P,{anchor:g.current,onDismiss:()=>h(!1),pinId:t,signals:e})]})}function w({pinKey:e}){let t=(0,o.Z)(j,e),n=(0,h.Z)(),{search:l}=(0,a.TH)(),r=(0,i.useMemo)(()=>new URLSearchParams(l),[l]);if(!n.isEmployee)return null;let{entityId:s,debAds:d,debContentQuality:c,debInclusiveProduct:u,debShopping:g,debTrustAndSafety:_}=t;if(d)return(0,p.jsx)(k,{pinId:s,signals:d});if(c)return(0,p.jsx)(k,{pinId:s,signals:c});if(u){if("skin_tone"===r.get("type")){let e=u.filter(e=>"visual_signals_hair_pattern_v1"!==e.signalId);return(0,p.jsx)(v,{pinId:s,signals:e})}{let e=u.filter(e=>"visual_body_analyses_v2_0"!==e.signalId);return(0,p.jsx)(v,{pinId:s,signals:e})}}return g?(0,p.jsx)(k,{pinId:s,signals:g}):_?(0,p.jsx)(k,{pinId:s,signals:_}):null}},43671:(e,t,n)=>{n.d(t,{ZP:()=>m});var l=n(667294),i=n(702664),a=n(883119),r=n(407043),s=n(898781),o=n(55275),d=n(680046),c=n(310227),u=n(841509),h=n(447948),p=n(785893);let g={background:{base:{transition:"all 250ms",opacity:0},blurred:{backgroundPosition:"50% 50%",backgroundSize:"cover",borderRadius:c.Oc,filter:"blur(10px)",opacity:1}},backdrop:{base:{transition:"all 250ms"},blurred:{backgroundColor:"rgba(0,0,0,0.6)"}},feedbackOverlayContainer:{base:{opacity:0,transition:"all 250ms"},visible:{opacity:1}}},_=["followed","related","search"],f=["pfy","pfyBoard"];function m({feedbackType:e,pin:t,hideContents:n=!1}){let m;let{logContextEvent:x}=(0,r.v)(),y=(0,s.ZP)(),[b,v]=(0,l.useState)(!1),[P,j]=(0,l.useState)(!1),k=(0,i.useDispatch)();(0,l.useEffect)(()=>v(!0),[]);let w=()=>j(!0),{id:S,feedbackText:D,images:C,showFeedbackOverlay:z}=t,A=C&&C["236x"]?.url,{subTitle:I,title:Z,unfollow:E,undoCallbackProps:F}=D??{},R=!_.includes(e),L=f.includes(e);if(F){let{action:e,actionOptions:t,viewType:n,viewParameter:l}=F;m=(0,d.EF)(t,e,S,n,l,e=>k((0,h.I1)(e)),x)}return(0,p.jsxs)(a.xu,{bottom:!0,color:"transparent",dangerouslySetInlineStyle:{__style:{borderRadius:c.Oc}},"data-test-id":"obscured-overlay",left:!0,overflow:"hidden",position:"absolute",right:!0,top:!0,children:[(0,p.jsxs)(a.xu,{bottom:!0,dangerouslySetInlineStyle:{__style:{borderRadius:c.Oc}},left:!0,margin:-4,overflow:"hidden",position:"absolute",right:!0,top:!0,children:[A&&(0,p.jsx)(u.Z,{additionalStyles:{...g.background.base,...b?g.background.blurred:{},backgroundImage:`url("${A}")`}}),(0,p.jsx)(u.Z,{additionalStyles:{...g.backdrop.base,...g.backdrop.blurred}})]}),(0,p.jsx)(a.xu,{dangerouslySetInlineStyle:{__style:{...g.feedbackOverlayContainer.base,...!z&&b?g.feedbackOverlayContainer.visible:{}}},height:"100%",padding:4,position:"relative",width:"100%",children:!n&&(0,p.jsx)(a.xu,{height:"100%",children:(0,p.jsxs)(a.xu,{"data-test-id":"PinFeedbackConfirmation",direction:"column",display:"flex",height:"100%",maxWidth:216,children:[(0,p.jsx)(a.xu,{marginBottom:2,children:(0,p.jsx)(a.X6,{color:"light",size:"400",children:Z})}),(0,p.jsx)(a.xu,{display:"flex",marginBottom:1,children:(0,p.jsx)(a.xv,{color:"light",overflow:"normal",children:I})}),(0,p.jsxs)(l.Fragment,{children:[m&&"Reported"!==Z&&(0,p.jsx)(a.xu,{"data-test-id":"undo-action-btn",children:(0,p.jsx)(a.iP,{onTap:m,children:(0,p.jsx)(a.xv,{color:"inverse",overflow:"normal",weight:"bold",children:y.bt("Undo", "Undo", "Text on the button to navigate to undo hiding a pin", undefined, true)})})}),(0,p.jsxs)(a.xu,{marginTop:"auto",children:[!m&&!P&&E&&!L&&(0,p.jsx)(a.zx,{color:"white",fullWidth:!0,onClick:()=>(0,o.t)(E.action,E.actionOptions,w,x),text:y.bt("Unfollow", "Unfollow", "Text on the button to unfollow a specific board / user", undefined, true)}),R&&(0,p.jsx)(a.ZP,{color:"white",fullWidth:!0,href:"/edit",onClick:()=>{x({event_type:101,view_type:1,view_parameter:92,element:11347})},text:y.bt("Tune your feed", "Tune your feed", "Text on the button to navigate to homefeed control", undefined, true)})]})]})]})})})]})}},202870:(e,t,n)=>{n.d(t,{Z:()=>d,j:()=>o});var l=n(883119),i=n(898781),a=n(349700),r=n(773285),s=n(785893);let o=[0,2,3];function d({hasAffiliateProducts:e,href:t,isPromoted:n,onNavigateSponsorClick:d,sponsorName:c,sponsorUsername:u,sponsorshipStatus:h,textWeight:p="bold"}){let g;let _=(0,i.ZP)(),f=(0,r.F)().checkExperiment("mweb_web_android_ios_clbc_eu_ad_string").anyEnabled,m=(0,r.F)().checkExperiment("web_remove_paid_partnership_in_rejected_state").anyEnabled,x=t||(u?`/${u}/`:null),y=x&&c?(0,s.jsx)(l.rU,{href:x,onClick:d,children:(0,s.jsx)(l.xv,{size:"200",weight:p,children:c})},c):void 0;return n?g=(0,a.nk)(_.bt("Promoted by {{ name }}", "Promoted by {{ name }}", "sponsorship.sponsorshipText.promotedByBrand", undefined, true),{name:y}):h||0===h?c&&!o.includes(h)?g=(0,a.nk)(_.bt("Paid partnership with {{ name }}", "Paid partnership with {{ name }}", "closeup.creator.sponsoredPinTitle", undefined, true),{name:y}):m&&2===h||(g=_.bt("Paid partnership", "Paid partnership", "closeup.creator.sponsoredPinTitle", undefined, true)):e&&(g=_.bt("Includes sponsored products", "Includes sponsored products", "sponsorship.sponsorshipText.affiliatedProducts", undefined, true)),(0,s.jsxs)(l.xv,{inline:!0,lineClamp:2,size:"200",children:[f&&!n&&"Ad • "||"",g]})}},966676:(e,t,n)=>{n.d(t,{Nb:()=>s,Of:()=>o,YO:()=>r,ZP:()=>u,x4:()=>c});var l=n(667294),i=n(499659),a=n(92261);let r=({showProductDetailPage:e,isLargerPane:t,showCloseupContentRight:n})=>e&&t?n?a.Uj:a.zT:1,s=(0,i.qe)(({paneWidth:e,pdpCarouselSlotToPaneRatio:t,showCloseupContentRight:n,showProductDetailPage:l,viewportSize:i,maxWidth:a,descriptionPaneWidth:r,isCloseupRelatedPinsAboveTheFoldEnabled:s,stackedCloseupEnabled:o,isInRemoveMagnifyingGlassVariant:d})=>({paneWidth:e,pdpCarouselSlotToPaneRatio:t,showCloseupContentRight:n,showProductDetailPage:l,viewportSize:i,maxWidth:a,descriptionPaneWidth:r,isCloseupRelatedPinsAboveTheFoldEnabled:s,stackedCloseupEnabled:o,isInRemoveMagnifyingGlassVariant:d})),o={showCloseupContentRight:!0,showProductDetailPage:!1,viewportSize:"lg",paneWidth:a.Gg,pdpCarouselSlotToPaneRatio:1,maxWidth:a.u6,descriptionPaneWidth:a.u6-a.Gg,isCloseupRelatedPinsAboveTheFoldEnabled:!1,stackedCloseupEnabled:!1,isInRemoveMagnifyingGlassVariant:!1},d=(0,l.createContext)(o);function c(){let e=(0,l.useContext)(d);if(!e)throw Error("useCloseupContext must be used within CloseupContextProvider");return e}let u=d},356725:(e,t,n)=>{n.r(t),n.d(t,{default:()=>f});var l,i,a=n(702664);n(167912);var r=n(883119),s=n(729884),o=n(916117),d=n(357998),c=n(966676),u=n(721782),h=n(447948),p=n(350118),g=n(785893);let _=void 0!==l?l:l=n(603239);function f({carouselData:e,carouselIndex:t,componentType:l,contextLogData:f,handleCarouselSwipe:m,isCloseup:x,isEditMode:y,pinKey:b,viewParameter:v,viewType:P,variant:j}){let k;let w=(0,p.S6)();if(null==b||"graphqlRef"===b.type)k=b;else{let e=b.data;if("string"==typeof e){let t=w(e);k=null!=t?{type:"deprecated",data:t}:null}else k={type:"deprecated",data:e}}let S=(0,o.Z)(_,k),{childDataKey__DEPRECATED:D}=(0,d.Q)(void 0!==i?i:i=n(822423),S,{useLegacyAdapter:e=>({})}),C=(0,u.Z)(D,"CarouselEllipsis"),z=(0,s.Z)(D),A=e||z&&{carouselSlots:z.carouselSlots.map(({slotId:e,title:t})=>({id:e,title:t})),entityId:z.carouselId??"",index:z.index},I=(0,a.useDispatch)();if(!A)return null;let Z=(e,t,n)=>{y||void 0===h.yR||(e.preventDefault(),e.stopPropagation(),I((0,h.yR)(S?.entityId??"",n))),void 0!==m&&m(n),C({pinId:S?.entityId??"",currentIndex:t??0,nextIndex:n,carouselData:{carouselSlots:A.carouselSlots?.map(e=>({id:e.id})),entityId:A.entityId},viewParameter:v,viewType:P,componentType:l,contextLogData:f,isEditMode:y})},{carouselSlots:E,index:F}=A,R="number"==typeof t?t:F,L="default"===j,T=L?"white":"#555",M=L?"rgba(255, 255, 255, 0.5)":"lightGray";return(0,g.jsx)(c.ZP.Consumer,{children:({viewportSize:e})=>"sm"===e?null:(0,g.jsx)(r.xu,{alignItems:"center","data-test-id":"carousel-ellipsis",display:"flex",justifyContent:x?"end":"center",paddingY:y?1:0,children:E&&[...Array(E.length??0).keys()].map(e=>(0,g.jsx)(r.xu,{paddingX:1,children:(0,g.jsx)(r.iP,{accessibilityLabel:(E[e]||{}).title??"",fullWidth:!1,onTap:({event:t})=>Z(t,R,e),rounding:"circle",children:(0,g.jsx)(r.xu,{dangerouslySetInlineStyle:{__style:{backgroundColor:e===R?T:M}},"data-test-id":"ellipsis-size",height:8,rounding:"circle",width:8})})},(S?.entityId??"")+e))})})}},721782:(e,t,n)=>{n.d(t,{Z:()=>d}),n(167912);var l,i=n(407043),a=n(916117),r=n(999018),s=n(67347);let o=void 0!==l?l:l=n(270643);function d(e,t){let{logContextEvent:n}=(0,i.v)(),l=(0,a.Z)(o,e);null!=e&&"deprecated"===e.type&&e.data&&"pin"!==e.data.type&&(0,s.nP)("web.graphql.debug.useLogSwipeError",{sampleRate:1,tags:{parent:t,rootType:e.data.type}});let d=(0,r.Z)(l);return function({pinId:e,currentIndex:t,nextIndex:l,carouselData:i,viewParameter:a,viewType:r,componentType:s,contextLogData:o,isEditMode:c,isEligibleForPdp:u}){if(!c){let{carouselSlots:c,entityId:h}=i;n({event_type:108,object_id_str:e,component:s,view_type:r,view_parameter:a,event_data:{pinCarouselSlotEventData:{carouselSlotId:c?.[t]&&Number(c[t].id),carouselDataId:Number(h),carouselSlotIndex:t,toCarouselSlotIndex:l}},aux_data:{...o,...d({isPdpPlus:u})}})}}}},436395:(e,t,n)=>{n.d(t,{Z:()=>a});var l=n(616550),i=n(784590);function a(){let{username:e}=(0,l.UO)();return(0,i.Z)(null==e?null:{name:"UserResource",options:{username:e,field_set_key:"profile"}})}}}]);
//# sourceMappingURL=https://sm.pinimg.com/webapp/34973.en_IN-0ff97eebd4b4ce81.mjs.map